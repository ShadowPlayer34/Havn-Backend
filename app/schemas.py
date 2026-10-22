"""Pydantic v2 contracts.

Three groups live here:

* ``Extracted*``  — what the LLM is contractually required to return.
* ``ThreatCheck*`` — the wire format the iOS client sees.
* ``ActiveThreat`` — the map/debug projection.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal


from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM extraction contract
# ---------------------------------------------------------------------------


class ThreatType(StrEnum):
    SHAHED = "shahed"
    MISSILE = "missile"
    RECON = "recon"


#: Ordering used when several threats cover the same point.
THREAT_PRIORITY: dict[str, int] = {
    ThreatType.MISSILE: 3,
    ThreatType.SHAHED: 2,
    ThreatType.RECON: 1,
}


# Implementation note (kept OUT of the docstring on purpose): these models are
# converted into Gemini's response schema, and a class docstring becomes the
# schema `description` — i.e. text the model reads as instruction. Commentary
# about Python internals belongs here, not there.
#
# They also carry no `extra="forbid"`: that makes Pydantic emit
# `additionalProperties`, which reaches the Gemini REST API as
# `additional_properties` and is rejected with a 400.
#
# `TargetRegion` is a list of pairs rather than a name -> region mapping for the
# same reason: a dict field is `additionalProperties` by another name.
class TargetRegion(BaseModel):
    """Where one of the target places lies."""

    sector: str = Field(
        description="The target_sectors entry this is about, copied exactly as "
        "written there."
    )
    region: str = Field(
        description="The oblast it lies in, or the city for a district, "
        'neighbourhood or landmark of a large city — e.g. "Київ", '
        '"Київська область", "Полтавська область".'
    )


class ExtractedThreat(BaseModel):
    """One aerial threat cluster: what it is, how many, and where it is headed."""

    threat_type: ThreatType = Field(
        description="Kind of aerial threat: shahed (one-way attack drone), "
        "missile (any cruise/ballistic/aeroballistic munition), or recon "
        "(reconnaissance UAV / observation aircraft)."
    )
    count: int = Field(
        ge=0,
        le=1000,
        description="Number of units. If several messages in the window give "
        "different numbers for the same group, report the MAXIMUM.",
    )
    target_sectors: list[str] = Field(
        default_factory=list,
        max_length=32,
        description="Administrative district / oblast / city names being "
        "threatened, exactly as written in the source. May also contain a raw "
        'coordinate pair formatted as "lat,lon" when the source gives one.',
    )
    target_regions: list[TargetRegion] = Field(
        default_factory=list,
        max_length=32,
        description="For each target_sectors entry smaller than an oblast, the "
        "oblast or large city it lies in. Omit entries you are unsure of, and "
        "oblasts and coordinates themselves.",
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How certain you are that this threat is real, correctly "
        "typed and correctly localised. Below 0.4 for vague or single-source "
        "rumours; above 0.8 only for explicit, corroborated reports.",
    )
    ttl_minutes: int = Field(
        default=20,
        ge=1,
        le=180,
        description="How long this threat should stay active before it "
        "auto-expires, in minutes. Default 20.",
    )
    eta_minutes: int | None = Field(
        default=None,
        ge=0,
        le=180,
        description="Estimated minutes until the threat reaches the target "
        "sectors, if the source states or clearly implies it. Null otherwise.",
    )
    origin_sector: str | None = Field(
        default=None,
        description="District/oblast the threat is approaching FROM, if "
        "stated. Used to draw the risk corridor. Null otherwise.",
    )
    source_refs: list[int] = Field(
        default_factory=list,
        max_length=64,
        description="Indices of the batch messages ([0], [1], ...) that "
        "support this threat. Never invent an index.",
    )

    def regions_by_sector(self) -> dict[str, str]:
        """`target_regions` keyed the way the resolver compares names."""
        return region_hints(self.target_regions)

    def dedup_key(self) -> str:
        """Stable identity for a cluster: type + the set of target sectors.

        Count and confidence are deliberately excluded — they are the fields
        that get merged when the same cluster is reported again.
        """
        sectors = sorted({normalize_sector(s) for s in self.target_sectors})
        payload = f"{self.threat_type}|{'|'.join(sectors)}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class ExtractionResult(BaseModel):
    """All active threat clusters described by one window of messages."""

    threats: list[ExtractedThreat] = Field(
        default_factory=list,
        max_length=32,
        description="Every distinct active threat cluster in this window. "
        "Merge duplicate reports of the same group into one entry. Empty list "
        "if the window contains no actionable threat information.",
    )
    all_clear_sectors: list[str] = Field(
        default_factory=list,
        max_length=64,
        description='Districts where an "all clear" / "відбій тривоги" was '
        "announced in this window. Active threats there will be expired "
        "immediately.",
    )


# ---------------------------------------------------------------------------
# Public API contract
# ---------------------------------------------------------------------------

RiskLevel = Literal["none", "elevated", "high"]


class ThreatCheckResponse(BaseModel):
    """Minimal payload for the iOS hot path. Kept deliberately tiny."""

    in_risk_zone: bool
    risk_level: RiskLevel
    threat_type: str | None = None
    eta_window: str | None = None
    #: The raw number `eta_window` is rendered from, decayed by neither this
    #: field nor the client. `eta_window` is a *snapshot* of the countdown at
    #: response time; a client polling every 30 s would otherwise watch the
    #: same string sit still and then jump. With this and the fact that it
    #: decays one minute per minute, the client can tick between polls — the
    #: same reason `/threats/zones` carries it raw per Feature.
    #:
    #: Null whenever no arrival time was extracted *or* estimated, which is
    #: also exactly when `eta_window` is null. The two are never inconsistent:
    #: both derive from the same stored value.
    eta_minutes: int | None = None
    sector_name: str | None = None
    #: The districts the threat was reported *against*, in the gazetteer's own
    #: spelling — the "Дарницький" in "Дарницький: летить дрон".
    #:
    #: Distinct from `sector_name`, which is where the *caller* is standing.
    #: The two differ precisely when the caller is in the buffer rather than
    #: the target, which is the `elevated` case: showing only `sector_name`
    #: there would name the user's own district as if it were under attack.
    #: A headline built from this field says what was actually reported, and
    #: stays stable for everyone looking at the same threat.
    target_sectors: list[str] = Field(default_factory=list)
    #: District the threat is approaching FROM — the origin of the risk
    #: corridor baked into `geometry`, not a second guess at it.
    #:
    #: Carries the *gazetteer's* name for that district, not the model's
    #: wording, and is null when the extracted origin matched no sector. That
    #: is deliberate: an unresolved origin drew no corridor, so showing it
    #: would put a place name on screen that nothing in the answer depends on.
    #: The raw extraction is still kept in `active_threats.raw` for audit.
    origin_sector: str | None = None
    #: Telegram message ids backing this threat, newest first, capped at
    #: `database.MAX_SOURCE_REFS`.
    #:
    #: Named for the client field it fills. These are *not* the LLM's
    #: `ExtractedThreat.source_refs`, which are indices into one batching
    #: window and are meaningless once that window is gone — they are the
    #: durable ids those indices were resolved to, so a number on screen can
    #: still be traced back to the message it came from.
    source_refs: list[int] = Field(default_factory=list)
    #: The `risk` polygon this answer was computed from, display-simplified, as
    #: a bare GeoJSON geometry object (`Polygon` / `MultiPolygon`).
    #:
    #: Null unless the caller passed `include_geometry=true`, and null whenever
    #: `in_risk_zone` is false. Opt-in because this response is `no-store` and
    #: polled every 30 s: a client that draws the whole map should hold the
    #: ETag-cached `/threats/zones` overlay instead and pay for the geometry
    #: once per snapshot. This field is for the callers that never fetch it —
    #: a widget or a Live Activity that needs one shape and no map.
    #:
    #: Typed loosely on purpose. GeoJSON is already a validated contract on the
    #: producing side (Shapely emits it), and a nested Pydantic model here
    #: would re-walk every coordinate on the latency-critical path to prove
    #: something that cannot be false.
    geometry: dict[str, Any] | None = None


class ActiveThreat(BaseModel):
    """Projection used by the map screen; not on the latency-critical path."""

    id: str
    threat_type: str
    count: int
    confidence_score: float
    #: Folded match keys ("дарницький"). Stable identity, not display text —
    #: render `target_sectors` instead.
    sectors: list[str]
    #: The same districts in the gazetteer's spelling ("Дарницький район").
    target_sectors: list[str] = Field(default_factory=list)
    expires_at: datetime
    updated_at: datetime
    eta_minutes: int | None = None


class ShelterOut(BaseModel):
    """One shelter, as the map needs it: identity, position, distance."""

    id: str
    name: str
    #: Raw OSM classification — "bunker", "bomb_shelter", "civil_defence", and so
    #: on. Nullable: a handful of objects carry no usable type tag. The client
    #: needs this to tell a bunker from a bus stop.
    shelter_type: str | None = None
    #: "kmda" (official register) or "osm". Trustworthiness differs; say so.
    source: str = "osm"
    address: str | None = None
    district: str | None = None
    #: The register's own wording for the structure ("Підвал з пандусом ОЗ").
    kind: str | None = None
    building_type: str | None = None
    #: "Цілодобово", or the restriction that applies.
    working_time: str | None = None
    phone: str | None = None
    accessible: bool | None = None
    notes: str | None = None
    #: Distinct devices that have reported a problem with this entry.
    reports: int = 0
    latitude: float
    longitude: float
    distance_meters: float


class NearestSheltersResponse(BaseModel):
    shelters: list[ShelterOut]


class ShelterReportRequest(BaseModel):
    """What a user can tell us about a shelter that is wrong on the map."""

    reason: Literal["wrong_location", "locked", "does_not_exist", "wrong_hours", "other"]
    note: str | None = Field(default=None, max_length=500)
    #: Client-generated UUID. Deduplicates one device's reports; not an identity.
    client_id: str = Field(min_length=8, max_length=64)


class ShelterReportAck(BaseModel):
    shelter_id: str
    reports: int


# ---------------------------------------------------------------------------
# Normalisation helpers (shared by extraction, storage and lookup)
# ---------------------------------------------------------------------------

# Administrative noise, in every case form the channels actually write. The
# inflected forms matter as much as the nominative: "Броварського району"
# scores 0.391 against "броварський" and is dropped, while the bare
# "броварський" is an exact hit — the only difference is this regex.
#
# The hromada type words ("міська"/"сільська"/"селищна") belong here too, and
# for a sharper reason: they are 88% of the gazetteer, and leaving them in
# name_norm both dilutes the trigram score of the settlement name and sends
# `load_sectors.derive_aliases` stemming the *type* instead of the place
# ("Шполянська міська громада" -> "шполянська мі"). Word-boundary guards keep
# them from biting into a name — the "ська" of "Львівська" is not preceded by
# a boundary, so it never matches.
_NOISE_RE = re.compile(
    r"(?<!\w)(?:"
    r"р[-‑]н|"
    r"район(?:у|і|ом|ів|и|ах|ами)?|"
    r"обл(?:\.|асть|асті|астю|астей)?|"
    r"громад(?:а|и|у|ою|і|ах)?|"
    r"територіальн(?:а|ої|ій|у)|"
    r"об(?:\'|’)?єднан(?:а|ої|ій|у)|"
    r"міськрад(?:а|и|і)|"
    r"(?:міськ|сільськ|селищн)(?:а|ої|ій|у|ого|ий|е)|"
    r"смт|с-ще|[мс]\.|"
    r"city|oblast|raion|district|hromada|rural|urban|settlement"
    r")(?!\w)",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

COORD_RE = re.compile(
    r"^\s*(-?\d{1,2}(?:[.,]\d+)?)\s*[,;]\s*(-?\d{1,3}(?:[.,]\d+)?)\s*$"
)


def normalize_sector(name: str) -> str:
    """Fold a free-text district name into a comparable key.

    Lowercases, strips administrative noise words and punctuation, and applies
    NFKC so that visually identical Cyrillic/Latin forms collapse together.
    """
    text = unicodedata.normalize("NFKC", name).casefold()
    text = _NOISE_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def region_hints(entries: list[TargetRegion]) -> dict[str, str]:
    """Region hints keyed on the folded sector name.

    Folded rather than raw so an entry still lines up when the model's echo of
    the sector differs only in case or punctuation from what it wrote in
    `target_sectors`. The first entry for a name wins.
    """
    regions: dict[str, str] = {}
    for entry in entries:
        key = normalize_sector(entry.sector)
        if key and entry.region.strip():
            regions.setdefault(key, entry.region)
    return regions


def parse_coordinate(value: str) -> tuple[float, float] | None:
    """Parse a ``"lat,lon"`` sector entry. Returns ``(lat, lon)`` or ``None``."""
    match = COORD_RE.match(value)
    if match is None:
        return None
    try:
        lat = float(match.group(1).replace(",", "."))
        lon = float(match.group(2).replace(",", "."))
    except ValueError:
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon
