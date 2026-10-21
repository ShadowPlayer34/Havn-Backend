"""The central extraction agent, backed by Google Gemini.

One LLM call per batching window, not per message — that is where the cost model
lives. The call is shaped so the stable part of the prompt comes first
(``system_instruction``) and only the batched messages vary, which is what lets
Gemini's implicit context caching discount the repeated prefix.

Structured output (``response_schema``) guarantees the response parses. There is
no regex extraction, no "output ONLY JSON" instruction, and no retry-on-parse
loop: the Pydantic model *is* the contract, and the SDK returns an instance of
it on ``response.parsed``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.schemas import ExtractionResult

logger = logging.getLogger(__name__)


# Provider-neutral failure modes. Everything below this line knows about
# Gemini; nothing above the ThreatExtractor interface does. Swapping providers
# again should touch this file only.


class LLMRefusal(RuntimeError):
    """Safety filters blocked the prompt or the response. Not retryable as-is."""


class LLMExtractionError(RuntimeError):
    """The model answered, but the answer was unusable."""


class LLMTransient(RuntimeError):
    """Rate limit, upstream 5xx or network failure. Losing the window is fine."""


class LLMRequestError(RuntimeError):
    """The request itself is wrong (bad model id, bad schema, bad key).

    Distinct from LLMTransient because retrying cannot fix it — this one should
    be loud in the logs, not a shrug.
    """


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

# Gemini accepts a *subset* of JSON Schema, and the SDK's automatic Pydantic
# conversion emits more than that subset — including `additional_properties`,
# which the REST API rejects outright with a 400. Rather than trust the
# conversion and debug opaque "Request contains an invalid argument" errors, we
# emit the schema ourselves and send only these keys.
_JSON_TO_GEMINI_TYPE = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def gemini_response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model into Gemini's response-schema dialect.

    Deliberately conservative: emits only ``type``, ``description``, ``enum``,
    ``items``, ``properties``, ``required`` and ``nullable``. Titles, defaults
    and numeric bounds are dropped — they are not reliably part of the accepted
    subset, and every constraint they express is still enforced client-side when
    the response is validated back into the Pydantic model.
    """
    raw = model.model_json_schema()
    return _convert_node(raw, raw.get("$defs", {}))


def _convert_node(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in node:
        ref_name = node["$ref"].rsplit("/", 1)[-1]
        merged = dict(defs[ref_name])
        # A `$ref` sibling description (Pydantic puts the Field description
        # next to the ref) should win over the referenced model's own.
        if "description" in node:
            merged["description"] = node["description"]
        node = merged

    nullable = False
    if "anyOf" in node:
        variants = [v for v in node["anyOf"] if v.get("type") != "null"]
        nullable = len(variants) < len(node["anyOf"])
        description = node.get("description")
        node = dict(variants[0]) if variants else {"type": "string"}
        if description:
            node.setdefault("description", description)

    out: dict[str, Any] = {}
    json_type = node.get("type")
    if json_type in _JSON_TO_GEMINI_TYPE:
        out["type"] = _JSON_TO_GEMINI_TYPE[json_type]
    if node.get("description"):
        out["description"] = node["description"]
    if node.get("enum"):
        out["enum"] = list(node["enum"])
        out.setdefault("type", "STRING")
    if nullable:
        out["nullable"] = True

    if json_type == "array":
        out["items"] = _convert_node(node.get("items", {"type": "string"}), defs)
    elif json_type == "object":
        properties = node.get("properties", {})
        out["properties"] = {
            name: _convert_node(sub, defs) for name, sub in properties.items()
        }
        # Require every property. Optional fields are expressed as `nullable`
        # instead, so the model must make an explicit decision about each one
        # rather than silently omitting it.
        out["required"] = list(properties)
    return out


#: Finish reasons that mean "a filter stopped this", as opposed to a bad request
#: or a truncated answer.
_BLOCKED_FINISH_REASONS = frozenset(
    {
        types.FinishReason.SAFETY,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.SPII,
        types.FinishReason.RECITATION,
    }
)

#: Categories worth relaxing for this workload. A civil-defence monitor is
#: *entirely* made of munitions vocabulary, and at the default thresholds
#: routine reports ("5 Shahed drones heading for Obolon") trip
#: DANGEROUS_CONTENT and come back empty — the system would go blind during
#: exactly the windows that matter. The threshold is configurable; see
#: LLM_SAFETY_THRESHOLD in .env.example for the trade-off.
_SAFETY_CATEGORIES = (
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the extraction stage of an air-raid threat-tracking system that warns \
civilians in Ukraine. You read raw, unstructured Telegram monitoring messages \
and turn them into structured threat records. Your output drives a map that \
people use to decide whether to take shelter, so both misses and false alarms \
carry real cost. Be accurate over dramatic.

# Input

You receive a batch of messages captured in one 30-60 second window. Each is \
labelled with an index like [0], [1], the source channel and a timestamp. \
Messages inside a window are usually redundant: several channels relay the same \
sighting, and one channel may post a correction seconds later.

# Your task

Produce the set of *distinct threat clusters* the window describes — not one \
record per message.

- Merge every message describing the same group of munitions over the same area \
  into a single threat.
- When merged messages disagree on the number of units, report the MAXIMUM.
- When a later message in the window corrects an earlier one (different \
  direction, different count, retraction), the later message wins.
- A message that only reports where a threat *was* previously, or reports past \
  impacts with no ongoing movement, is not an active threat. Skip it.
- Historical summaries, fundraising posts, political commentary, casualty \
  reports and equipment news are not threats. Skip them.

# threat_type

- `shahed`  — one-way attack drones: Shahed-131/136, Geran-1/2, "мопед", \
  "шахед", "герань", "БпЛА" used in a strike role.
- `missile` — any missile: cruise (Kalibr/Калібр, Kh-101/Х-101, Kh-555), \
  ballistic (Iskander/Іскандер, KN-23), aeroballistic (Kinzhal/Кинджал), \
  anti-ship (Onyx/Онікс), or S-300/S-400 used against ground targets.
- `recon`   — reconnaissance and observation only: Orlan/Орлан, ZALA, Merlin, \
  Supercam, and unarmed observation flights.

If a message describes a carrier aircraft that has not yet launched (e.g. Tu-95 \
takeoff, MiG-31K airborne), treat it as `missile` with a low `confidence_score` \
and a longer `ttl_minutes`, since the launch has not happened yet.

# target_sectors

Copy district, city or oblast names as the source writes them — do not \
translate, do not expand abbreviations, do not guess a parent region that was \
not mentioned. The downstream resolver handles inflection and spelling. If the \
source gives explicit coordinates instead of a place name, emit them as a \
single string `"lat,lon"` (e.g. `"50.4501,30.5234"`).

Leave `target_sectors` empty rather than inventing a location. An unlocalised \
threat is dropped downstream; a wrongly localised one sends people to shelter \
in the wrong city.

# confidence_score

- 0.85-1.00 — explicit, specific, corroborated by two or more messages.
- 0.60-0.84 — one clear, specific report from a reliable monitoring channel.
- 0.40-0.59 — vague direction or ambiguous location; single source.
- 0.00-0.39 — rumour, speculation, unconfirmed, or you are unsure of the type.

# ttl_minutes

How long the threat stays actionable. Default 20. Use a shorter value for \
fast-moving munitions already near their target, and a longer value for \
carriers that have taken off but not launched, or for slow drone groups \
crossing a large distance.

# all_clear_sectors

If the window contains an all-clear ("відбій тривоги", "відбій загрози"), list \
the districts it covers. Active threats there are retired immediately.

# Grounding

Every threat must cite the batch indices that support it in `source_refs`. \
Never cite an index that is not in the batch. If the window contains no \
actionable threat information, return an empty `threats` list — that is a \
correct and expected answer, and most windows will look like that.
"""


@dataclass(slots=True)
class BatchMessage:
    channel: str
    message_id: int
    timestamp: Any  # datetime; kept loose so this module stays import-light
    text: str


def _batch_to_text(messages: list[BatchMessage]) -> str:
    lines = ["Messages captured in this window:", ""]
    for index, message in enumerate(messages):
        stamp = message.timestamp.strftime("%H:%M:%S")
        lines.append(f"[{index}] {stamp} @{message.channel}")
        lines.append(message.text.strip())
        lines.append("")
    lines.append("Extract the distinct active threat clusters described above.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class ThreatExtractor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._config = self._build_config()
        self._client: genai.Client | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self._settings.gemini_api_key)

    def _get_client(self) -> genai.Client:
        """Build the client on first use, not at import/startup.

        The SDK raises immediately when constructed without a key. Doing that
        eagerly would take the whole worker down — janitor, TTL expiry and
        snapshot publishing included — over a credential only the extraction
        step needs. Lazily, a missing or rotated key degrades exactly one
        stage and the rest keeps running.
        """
        if self._client is None:
            if not self.is_configured:
                raise LLMRequestError("GEMINI_API_KEY is not set")
            self._client = genai.Client(
                api_key=self._settings.gemini_api_key,
                http_options=types.HttpOptions(
                    # NOTE: Gemini expresses this in milliseconds, unlike most
                    # Python HTTP clients.
                    timeout=int(self._settings.llm_timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(
                        attempts=self._settings.llm_max_retries,
                    ),
                ),
            )
        return self._client

    async def aclose(self) -> None:
        # The SDK owns its transport and exposes no explicit close; this exists
        # so callers do not need to care which provider is behind the interface.
        return None

    def _build_config(self) -> types.GenerateContentConfig:
        threshold = types.HarmBlockThreshold(self._settings.llm_safety_threshold)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            # Our own conversion rather than passing the Pydantic class — see
            # gemini_response_schema(). The field descriptions carried through
            # are real instructions to the model, so schemas.py doubles as
            # prompt surface.
            response_schema=gemini_response_schema(ExtractionResult),
            max_output_tokens=self._settings.llm_max_tokens,
            # We declare no tools, but the SDK still initialises its automatic
            # function-calling path and warns about using it outside a chat
            # session. Disabling it removes that warning from every window.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            safety_settings=[
                types.SafetySetting(category=category, threshold=threshold)
                for category in _SAFETY_CATEGORIES
            ],
        )
        # Omitted entirely when unset, so the same code path works on models
        # that do not accept a thinking level (e.g. the 2.5 family, which uses
        # thinking_budget instead).
        if self._settings.llm_thinking_level:
            config.thinking_config = types.ThinkingConfig(
                thinking_level=types.ThinkingLevel(
                    self._settings.llm_thinking_level.upper()
                )
            )
        return config

    async def extract(self, messages: list[BatchMessage]) -> ExtractionResult:
        """Run one window through the model. Raises on refusal or bad output."""
        if not messages:
            return ExtractionResult()

        client = self._get_client()
        try:
            response = await client.aio.models.generate_content(
                model=self._settings.llm_model,
                contents=_batch_to_text(messages),
                config=self._config,
            )
        except genai_errors.ClientError as exc:
            # 429 is expected under load; every other 4xx means the request is
            # malformed and no amount of retrying will help.
            if exc.code == 429:
                raise LLMTransient(f"rate limited: {exc}") from exc
            raise LLMRequestError(f"request rejected ({exc.code}): {exc}") from exc
        except genai_errors.ServerError as exc:
            raise LLMTransient(f"upstream error ({exc.code}): {exc}") from exc

        # Order matters: a blocked prompt produces a response with no candidates
        # at all, so check prompt feedback before reaching for candidates, and
        # candidates before reaching for text.
        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason is not None:
            raise LLMRefusal(f"prompt blocked (reason={feedback.block_reason})")

        if not response.candidates:
            raise LLMExtractionError("response contained no candidates")

        candidate = response.candidates[0]
        if candidate.finish_reason in _BLOCKED_FINISH_REASONS:
            raise LLMRefusal(f"response blocked (finish_reason={candidate.finish_reason})")
        if candidate.finish_reason == types.FinishReason.MAX_TOKENS:
            # The JSON is truncated and unparseable. Raising is honest; raise
            # LLM_MAX_TOKENS if this shows up in the logs.
            raise LLMExtractionError("hit max_output_tokens — JSON truncated")

        usage = response.usage_metadata
        if usage is not None:
            logger.info(
                "llm_window model=%s messages=%d in=%d out=%d cached=%d thoughts=%d",
                self._settings.llm_model,
                len(messages),
                usage.prompt_token_count or 0,
                usage.candidates_token_count or 0,
                usage.cached_content_token_count or 0,
                usage.thoughts_token_count or 0,
            )

        # `parsed` is the schema-validated instance. It can still be None if the
        # model emitted JSON the schema could not bind, so fall back to parsing
        # the raw text before giving up.
        result = response.parsed
        if isinstance(result, ExtractionResult):
            return result

        if not response.text:
            raise LLMExtractionError(
                f"empty response body (finish_reason={candidate.finish_reason})"
            )
        try:
            return ExtractionResult.model_validate_json(response.text)
        except ValidationError as exc:
            raise LLMExtractionError(f"response failed validation: {exc}") from exc
