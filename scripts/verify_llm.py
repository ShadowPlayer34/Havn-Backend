"""Verify the Gemini extraction path with a single API call.

    docker compose exec worker python scripts/verify_llm.py

Sends one realistic batch — two channels relaying the same drone group, a
correction, a fundraising post that must be ignored, and an all-clear — and
prints what came back. Deliberately one call: free-tier quota is easy to spend.

If it fails with a 400 INVALID_ARGUMENT (the API's unhelpful catch-all for a
rejected schema), re-run with --bisect to find which field it dislikes:

    docker compose exec worker python scripts/verify_llm.py --bisect

That costs roughly one call per field, so only reach for it when you need it.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime

from pydantic import BaseModel, create_model

from app.config import get_settings
from app.llm import (
    BatchMessage,
    LLMExtractionError,
    LLMRefusal,
    LLMRequestError,
    LLMTransient,
    ThreatExtractor,
    gemini_response_schema,
)
from app.schemas import ExtractedThreat, ExtractionResult

BATCH = [
    ("AerisRimor", "Шахеди курсом на Оболонь, 5 шт. Орієнтовний час підльоту 10 хвилин."),
    ("radar_tryzub", "БпЛА в напрямку Києва з боку Київської області, близько 7 одиниць"),
    ("AerisRimor", "Уточнення: група БпЛА 7 шт, курс на Оболонський район"),
    ("radar_tryzub", "Друзі, підтримайте наш збір на РЕБ для 47 бригади!"),
    ("AerisRimor", "Відбій повітряної тривоги в Одеській області"),
]


async def verify() -> int:
    settings = get_settings()
    extractor = ThreatExtractor(settings)
    now = datetime.now(UTC)
    batch = [
        BatchMessage(channel, i, now, text) for i, (channel, text) in enumerate(BATCH)
    ]

    print(
        f"model={settings.llm_model}  thinking={settings.llm_thinking_level or '(off)'}  "
        f"safety={settings.llm_safety_threshold}\n"
    )
    try:
        result = await extractor.extract(batch)
    except LLMRefusal as exc:
        print(f"REFUSED — {exc}")
        print("  Safety filters blocked realistic threat text.")
        print("  Loosen LLM_SAFETY_THRESHOLD (BLOCK_NONE, or OFF) and retry.")
        return 1
    except LLMRequestError as exc:
        print(f"REQUEST REJECTED — {exc}")
        print("  400 = schema rejected (try --bisect). 404 = model id retired or")
        print("  unavailable to this key. 403 = key lacks access.")
        return 1
    except LLMTransient as exc:
        print(f"TRANSIENT — {exc}")
        print("  429 = quota spent; free-tier daily limits reset on Pacific midnight.")
        return 1
    except LLMExtractionError as exc:
        print(f"BAD OUTPUT — {exc}")
        return 1

    print(f"SUCCESS — {len(result.threats)} threat(s) extracted\n")
    for threat in result.threats:
        print(f"  {threat.threat_type}  count={threat.count}")
        print(f"    sectors    : {threat.target_sectors}")
        print(f"    confidence : {threat.confidence_score}   ttl: {threat.ttl_minutes}m")
        print(f"    eta        : {threat.eta_minutes}   origin: {threat.origin_sector}")
        print(f"    grounded in: messages {threat.source_refs}")
    print(f"\n  all_clear_sectors: {result.all_clear_sectors}")

    print("\nWhat to sanity-check in the output above:")
    print("  - the two drone reports merged into ONE threat, count 7 (the maximum)")
    print("  - the fundraising post produced nothing")
    print("  - Одеська appears in all_clear_sectors, not as a threat")
    return 0


async def bisect() -> int:
    """Add ExtractedThreat's fields back one at a time to find the rejected one."""
    from google import genai
    from google.genai import types

    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)
    fields = list(ExtractedThreat.model_fields.items())

    print(f"model={settings.llm_model} — testing {len(fields)} field(s) cumulatively\n")
    for count in range(1, len(fields) + 1):
        subset = dict(fields[:count])
        partial: type[BaseModel] = create_model(  # type: ignore[call-overload]
            "Partial",
            **{name: (f.annotation, f) for name, f in subset.items()},
        )
        wrapper: type[BaseModel] = create_model(
            "Wrapper", threats=(list[partial], ...)  # type: ignore[valid-type]
        )
        newest = list(subset)[-1]
        try:
            await client.aio.models.generate_content(
                model=settings.llm_model,
                contents="Шахеди курсом на Оболонь, 5 шт",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=gemini_response_schema(wrapper),
                ),
            )
            print(f"  OK   +{newest}")
        except Exception as exc:  # noqa: BLE001 — diagnostic script
            text = str(exc).replace("\n", " ")
            print(f"  FAIL +{newest}: {text[:140]}")
            if "429" not in text:
                print(f"\n  => '{newest}' is the field the API rejects.")
            return 1
    print("\n  => full schema accepted; the problem is elsewhere in the request.")
    return 0


if __name__ == "__main__":
    coro = bisect() if "--bisect" in sys.argv else verify()
    raise SystemExit(asyncio.run(coro))
