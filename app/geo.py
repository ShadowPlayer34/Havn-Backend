"""Turning free-text sector names into PostGIS risk geometry.

The LLM emits district names the way a Telegram channel writes them — inflected,
abbreviated, sometimes misspelled. Resolution is a three-step ladder:

1. exact match on the normalised name,
2. membership in the sector's ``aliases`` array,
3. ``pg_trgm`` similarity above a configured threshold.

All geometry math happens in PostGIS on the ``geography`` type, so buffer
distances are real metres rather than degrees (a 15 km buffer must not become
an ellipse that is 40% too narrow at Ukrainian latitudes).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import TextClause, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import normalize_sector, parse_coordinate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResolvedSectors:
    sector_ids: list[int] = field(default_factory=list)
    matched_names: list[str] = field(default_factory=list)
    normalized_names: list[str] = field(default_factory=list)
    points: list[tuple[float, float]] = field(default_factory=list)  # (lat, lon)
    unmatched: list[str] = field(default_factory=list)
    #: Unmatched because the only match was a guess far from the rest of the
    #: report. Also listed in `unmatched`.
    too_far: list[str] = field(default_factory=list)
    #: Placed by the model's region hint alone: nothing by that name inside the
    #: region, so the region itself stands in for it.
    region_fallbacks: list[str] = field(default_factory=list)

    @property
    def is_localisable(self) -> bool:
        return bool(self.sector_ids or self.points)


# Tie-breaking, in both queries, in this order:
#
# 1. An exact name beats an alias. "київ" is Kyiv the city, even though it is
#    also a derived alias of Київська область.
# 2. Proximity to the rest of the report. Forty hromada names are not unique —
#    "Кам'янська міська" is both Кам'янське on the Dnipro and Кам'янка in
#    Cherkasy oblast — and area alone chose between them by coin flip: a report
#    reading "Кам'янське, Дніпро" drew its zone 161 km from the city it named.
#    Sectors that resolved to a single candidate anchor the ones that did not.
#    With no anchor the term is constant and this degrades to the rules below.
# 3. The narrower administrative *type*. A bare settlement name means the
#    settlement: without this "Чорноморськ" resolved to a 344 km² rural hromada
#    two oblasts away rather than the city of that name. Raions and oblasts
#    rank with міська rather than below it, so an oblast-level report is still
#    read as oblast-level.
# 4. Area, descending — the original rule, and still what applies when nothing
#    above separates the candidates. It over-warns rather than under-warns,
#    which is the correct direction to fail for a shelter alert.
#
# `candidates` counts the rows that tied at the top, which is what makes a
# match usable as an anchor: one candidate is a fact, several are a guess.
_TYPE_RANK = """
    CASE WHEN {t}.name LIKE '%міська громада%' THEN 0
         WHEN {t}.name LIKE '%селищна громада%' THEN 1
         WHEN {t}.name LIKE '%сільська громада%' THEN 2
         ELSE 0
    END
"""

# NULL centroids sort last under ASC, so a sector without one loses the
# proximity tie-break rather than winning it.
_ANCHOR_DISTANCE = """
    CASE WHEN CAST(:anchor AS text) IS NULL THEN 0
         ELSE ST_Distance({t}.centroid::geography,
                          ST_GeogFromText(CAST(:anchor AS text)))
    END
"""

# A region hint from the model narrows a lookup to the sectors inside that
# oblast or city. Holes are filled first: Kyiv is a hole in Київська область,
# and a hint of "Київська область" must still admit the city it surrounds.
# Union rather than collect, because a filled part can swallow an enclave that
# is a part of its own.
# `ST_PointOnSurface` rather than the stored centroid because a crescent-shaped
# hromada's centroid can fall outside it, and with it outside its oblast.
_REGION_CTE = """
    WITH region AS (
        SELECT ST_SetSRID(ST_Union(ARRAY(
                   SELECT ST_MakePolygon(ST_ExteriorRing(d.geom))
                   FROM ST_Dump(r.geom) AS d
               )), 4326) AS shell
        FROM sectors r
        WHERE r.id = :region_id
    )
"""
_IN_REGION = "ST_Covers((SELECT shell FROM region), ST_PointOnSurface({t}.geom))"


def _match_sql(in_region: bool) -> TextClause:
    return text(
        f"""
        {_REGION_CTE if in_region else ""}
        SELECT s.id, s.name, s.name_norm,
               ST_Y(s.centroid) AS lat, ST_X(s.centroid) AS lon,
               count(*) OVER (PARTITION BY (s.name_norm = :norm)) AS candidates,
               (s.name_norm = :norm) AS exact
        FROM sectors s
        WHERE (s.name_norm = :norm OR :norm = ANY(s.aliases))
          {"AND " + _IN_REGION.format(t="s") if in_region else ""}
        ORDER BY (s.name_norm = :norm) DESC,
                 {_ANCHOR_DISTANCE.format(t="s")} ASC,
                 {_TYPE_RANK.format(t="s")} ASC,
                 ST_Area(s.geom) DESC
        LIMIT 1
        """
    )


# Scores against aliases as well as the canonical name. Without this the
# derived stems only ever produce exact hits, and "Оболонь" (0.667 against the
# stem "оболон", 0.429 against "оболонський") stays unmatched.
#
# This scans the table rather than using the trigram index. That is deliberate:
# the gazetteer is ~1600 rows and resolution runs once per extracted threat in
# the worker, never on the API hot path.
def _fuzzy_sql(in_region: bool) -> TextClause:
    return text(
        f"""
        {_REGION_CTE if in_region else ""}
        SELECT s.id, s.name, s.name_norm, scored.score,
               ST_Y(s.centroid) AS lat, ST_X(s.centroid) AS lon,
               count(*) OVER (PARTITION BY scored.score) AS candidates,
               false AS exact
        FROM sectors s,
             LATERAL (
                 SELECT GREATEST(
                     similarity(s.name_norm, :norm),
                     COALESCE(
                         (SELECT max(similarity(alias, :norm)) FROM unnest(s.aliases) AS alias),
                         0
                     )
                 ) AS score
             ) AS scored
        WHERE scored.score >= :threshold
          {"AND " + _IN_REGION.format(t="s") if in_region else ""}
        ORDER BY scored.score DESC,
                 {_ANCHOR_DISTANCE.format(t="s")} ASC,
                 {_TYPE_RANK.format(t="s")} ASC,
                 ST_Area(s.geom) DESC
        LIMIT 1
        """
    )


_MATCH_SQL = _match_sql(in_region=False)
_FUZZY_SQL = _fuzzy_sql(in_region=False)
_MATCH_IN_REGION_SQL = _match_sql(in_region=True)
_FUZZY_IN_REGION_SQL = _fuzzy_sql(in_region=True)

_COVERED_BY_REGION_SQL = text(
    f"""
    {_REGION_CTE}
    SELECT EXISTS (
               SELECT 1 FROM sectors s
               WHERE s.id = ANY(CAST(:sector_ids AS int[]))
                 AND {_IN_REGION.format(t="s")}
           )
        OR EXISTS (
               SELECT 1
               FROM unnest(CAST(:lats AS float8[]), CAST(:lons AS float8[])) AS p(lat, lon)
               WHERE ST_Covers((SELECT shell FROM region),
                               ST_SetSRID(ST_MakePoint(p.lon, p.lat), 4326))
           )
    """
)

_DISTANCE_TO_ANCHOR_SQL = text(
    """
    SELECT ST_Distance(ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography,
                       ST_GeogFromText(CAST(:anchor AS text)))
    """
)


def _anchor_wkt(points: list[tuple[float, float]]) -> str | None:
    """Everything in this report whose position is already certain."""
    if not points:
        return None
    coords = ", ".join(f"{lon} {lat}" for lat, lon in points)
    return f"MULTIPOINT({coords})"


@dataclass(slots=True)
class _Lookup:
    """One name on its way through `resolve_sectors`."""

    raw: str
    norm: str
    row: Any | None
    #: Matched on trigram similarity alone — a guess, not a fact.
    fuzzy: bool = False
    #: The sector the model said this place lies in, when that resolved.
    region: Any | None = None

    def certain(self) -> bool:
        """Good enough to tell a far guess in the same report it is wrong.

        A gazetteer name spelled out in full, or anything the model placed in
        a region. Not an alias: the derived stems are exactly what collides —
        "буча" is a stem of Бучацька in Ternopil oblast, not of Бучанська, and
        measuring "Ірпінь" against it would drop the one correct match in the
        report.
        """
        return (
            self.row is not None
            and self.row["candidates"] == 1
            and self.row["lat"] is not None
            and (self.row["exact"] or self.region is not None)
        )


async def _lookup(
    session: AsyncSession,
    norm: str,
    threshold: float,
    anchor: str | None,
    region_id: int | None = None,
) -> tuple[Any | None, bool]:
    """One climb up the ladder: exact name, then alias, then trigram.

    Returns the row and whether it took the trigram step to find it.
    """
    params: dict[str, Any] = {"norm": norm, "anchor": anchor}
    exact_sql, fuzzy_sql = _MATCH_SQL, _FUZZY_SQL
    if region_id is not None:
        params["region_id"] = region_id
        exact_sql, fuzzy_sql = _MATCH_IN_REGION_SQL, _FUZZY_IN_REGION_SQL
    row = (await session.execute(exact_sql, params)).mappings().first()
    if row is not None:
        return row, False
    row = (
        (await session.execute(fuzzy_sql, {**params, "threshold": threshold}))
        .mappings()
        .first()
    )
    return row, row is not None


async def resolve_sectors(
    session: AsyncSession,
    raw_names: list[str],
    threshold: float,
    regions: Mapping[str, str] | None = None,
    fuzzy_max_meters: float | None = None,
    region_threshold: float | None = None,
) -> ResolvedSectors:
    """Map raw sector strings onto gazetteer rows and/or bare coordinates.

    Two passes. The first resolves every name on its own and records how many
    candidates tied for the win; the second re-resolves only the names that had
    a real choice, this time near the ones that did not. A single name in,
    a single name out — `resolve_origin` relies on that — but a report naming
    several places disambiguates itself.

    `regions` maps a folded name (`normalize_sector`) to the oblast or city the
    model says it lies in. A hinted name is only ever matched inside that
    region; the hint is a filter, because the names it exists for — Поділ,
    Воскресенка, "Правий берег" — have a better-scoring namesake elsewhere.
    Inside a region the trigram step uses `region_threshold` (default:
    `threshold`); with a few dozen candidates instead of 1600, a lower bar
    lets inflected forms through — "Бучу" scores 0.38 against Бучанська.

    `fuzzy_max_meters` drops trigram matches on unhinted names that land
    further than that from everything else in the same report.
    """
    resolved = ResolvedSectors()
    pending: list[_Lookup] = []
    region_rows: dict[str, Any | None] = {}
    if region_threshold is None:
        region_threshold = threshold

    for raw in raw_names:
        if not raw or not raw.strip():
            continue

        coord = parse_coordinate(raw)
        if coord is not None:
            resolved.points.append(coord)
            resolved.normalized_names.append(f"{coord[0]:.4f},{coord[1]:.4f}")
            continue

        norm = normalize_sector(raw)
        if not norm:
            continue

        region = await _region_row(session, (regions or {}).get(norm), threshold, region_rows)
        if region is not None:
            row, fuzzy = await _lookup(session, norm, region_threshold, None, region["id"])
        else:
            row, fuzzy = await _lookup(session, norm, threshold, None)
        pending.append(_Lookup(raw, norm, row, fuzzy, region))

    # A homonym is only as good as the company it keeps. Unambiguous matches —
    # and any raw coordinates — say where this report is.
    anchor = _anchor_wkt(
        [
            (item.row["lat"], item.row["lon"])
            for item in pending
            if item.row is not None
            and item.row["candidates"] == 1
            and item.row["lat"] is not None
        ]
        + resolved.points
    )

    for item in pending:
        if anchor is not None and item.row is not None and item.row["candidates"] > 1:
            # Keep the context-free match if the second pass finds nothing;
            # an anchor is a preference, never a filter.
            if item.region is not None:
                row, fuzzy = await _lookup(
                    session, item.norm, region_threshold, anchor, item.region["id"]
                )
            else:
                row, fuzzy = await _lookup(session, item.norm, threshold, anchor)
            if row is not None:
                item.row, item.fuzzy = row, fuzzy

    if fuzzy_max_meters is not None:
        await _drop_distant_guesses(session, pending, resolved, fuzzy_max_meters)
    await _fall_back_to_regions(session, pending, resolved)

    seen_ids: set[int] = set()
    for item in pending:
        if item.row is None:
            resolved.unmatched.append(item.raw)
            logger.info("sector_unmatched name=%r normalized=%r", item.raw, item.norm)
            continue

        if item.row["id"] in seen_ids:
            continue
        seen_ids.add(item.row["id"])
        resolved.sector_ids.append(item.row["id"])
        resolved.matched_names.append(item.row["name"])
        resolved.normalized_names.append(item.row["name_norm"])

    return resolved


async def _region_row(
    session: AsyncSession,
    name: str | None,
    threshold: float,
    cache: dict[str, Any | None],
) -> Any | None:
    """The sector a region hint names, or None when there is no usable hint.

    A hint that resolves to nothing is dropped rather than trusted: the name it
    was attached to falls back to the unhinted ladder.
    """
    if not name:
        return None
    norm = normalize_sector(name)
    if not norm:
        return None
    if norm not in cache:
        row, _ = await _lookup(session, norm, threshold, None)
        cache[norm] = row
        if row is None:
            logger.info("region_unmatched region=%r", name)
    return cache[norm]


async def _drop_distant_guesses(
    session: AsyncSession,
    pending: list[_Lookup],
    resolved: ResolvedSectors,
    max_meters: float,
) -> None:
    """Unmatch trigram guesses that land far from the rest of their report.

    "Правий берег" on a Kyiv report scores 0.46 against Берегове's derived stem
    "берег", and the anchor tie-break cannot help when both candidates are in
    Zakarpattia. Replayed over the recorded history, correct guesses sit within
    ~250 km of the rest of their report and wrong ones from there out to 800 —
    see `sector_fuzzy_max_km`.

    Only unhinted fuzzy matches are judged, and only in a report with something
    certain in it — otherwise there is nothing to say which side is wrong. Each
    is measured against every other place the report resolved, not just the
    certain ones: in "Кропивницький, Знам'янка, Летичів, Хмельниччині" the only
    certain place is 330 km from Хмельницький, but Летичів, a guess itself, is
    50 km from it and the two corroborate each other.
    """
    if not resolved.points and not any(item.certain() for item in pending):
        return
    for item in pending:
        if item.row is None or not item.fuzzy or item.region is not None:
            continue
        if item.row["lat"] is None:
            continue
        # Oblasts are exempt. Their names are distinctive enough that a guess
        # landing on one is nearly always right, the model gives them no region
        # by design, and an oblast's centroid sits hundreds of km from the city
        # it is reported alongside: "Волинь" in a ten-oblast report is 406 km
        # from Київ.
        if item.row["name"].endswith("область"):
            continue
        anchor = _anchor_wkt(
            [
                (other.row["lat"], other.row["lon"])
                for other in pending
                if other is not item and other.row is not None and other.row["lat"] is not None
            ]
            + resolved.points
        )
        if anchor is None:
            continue
        distance = (
            await session.execute(
                _DISTANCE_TO_ANCHOR_SQL,
                {"lat": item.row["lat"], "lon": item.row["lon"], "anchor": anchor},
            )
        ).scalar_one()
        if distance > max_meters:
            logger.info(
                "sector_too_far name=%r matched=%r km=%.0f",
                item.raw,
                item.row["name"],
                distance / 1000.0,
            )
            resolved.too_far.append(item.raw)
            item.row = None


async def _fall_back_to_regions(
    session: AsyncSession, pending: list[_Lookup], resolved: ResolvedSectors
) -> None:
    """Give a hinted name with no match inside its region the region itself.

    Only when nothing else in the report already lies in that region: an
    unmatched "Правий берег" next to a matched "Київ" adds nothing, while the
    same name alone still warns Kyiv rather than nobody — over-warning inside
    the right city is the correct way for a shelter alert to fail.
    """
    for item in pending:
        if item.row is not None or item.region is None:
            continue
        others = [
            other.row["id"]
            for other in pending
            if other is not item and other.row is not None
        ]
        covered = (
            await session.execute(
                _COVERED_BY_REGION_SQL,
                {
                    "region_id": item.region["id"],
                    "sector_ids": others,
                    "lats": [lat for lat, _ in resolved.points],
                    "lons": [lon for _, lon in resolved.points],
                },
            )
        ).scalar_one()
        if covered:
            continue
        logger.info(
            "sector_region_fallback name=%r region=%r", item.raw, item.region["name"]
        )
        resolved.region_fallbacks.append(item.raw)
        item.row = item.region


@dataclass(slots=True)
class ResolvedOrigin:
    """The approach sector, once a real gazetteer row has been found for it."""

    sector_id: int
    #: The gazetteer's canonical spelling, not the model's. This is what the
    #: client shows as "approaching from", so it has to be a name that exists.
    name: str


async def resolve_origin(
    session: AsyncSession, raw_name: str | None, threshold: float
) -> ResolvedOrigin | None:
    """Resolve the corridor origin. Coordinates are not accepted here.

    Returns the name alongside the id because both ends of the corridor are
    user-visible: the id draws it, the name explains it. Failing to resolve
    means no corridor *and* nothing to display — the two go together, which is
    why one return value covers both.
    """
    if not raw_name:
        return None
    resolved = await resolve_sectors(session, [raw_name], threshold)
    if not resolved.sector_ids:
        return None
    # `sector_ids` and `matched_names` are appended in lockstep; coordinates
    # only ever land in `points`, so index 0 lines up.
    return ResolvedOrigin(sector_id=resolved.sector_ids[0], name=resolved.matched_names[0])


_GEOMETRY_SQL = text(
    """
    WITH sector_geoms AS (
        SELECT geom FROM sectors WHERE id = ANY(CAST(:sector_ids AS int[]))
    ),
    point_geoms AS (
        SELECT ST_Buffer(
                   ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography,
                   :point_radius
               )::geometry AS geom
        FROM unnest(CAST(:lats AS float8[]), CAST(:lons AS float8[])) AS t(lat, lon)
    ),
    core AS (
        SELECT ST_UnaryUnion(ST_Collect(geom)) AS g
        FROM (SELECT geom FROM sector_geoms UNION ALL SELECT geom FROM point_geoms) u
    ),
    buf AS (
        -- The margin scales with the target: sqrt(area) is the sector's
        -- characteristic width in metres, so `:buffer_ratio` reads as a
        -- fraction of it. A flat distance cannot serve both a 20 km² city
        -- district and a 41 000 km² oblast; this does, and the clamp keeps
        -- both ends honest.
        SELECT LEAST(
                   CAST(:buffer_ceiling AS float8),
                   GREATEST(
                       CAST(:buffer_floor AS float8),
                       CAST(:buffer_ratio AS float8) * sqrt(ST_Area(core.g::geography))
                   )
               ) AS m
        FROM core
    ),
    corridor AS (
        -- Directional risk corridor: everything within `buffer` metres of the
        -- straight line from the approach sector to the target's centroid.
        SELECT CASE
                   WHEN o.centroid IS NULL OR core.g IS NULL THEN NULL
                   ELSE ST_Buffer(
                            ST_MakeLine(o.centroid, ST_Centroid(core.g))::geography,
                            buf.m
                        )::geometry
               END AS g
        FROM core, buf
        LEFT JOIN sectors o ON o.id = CAST(:origin_id AS int)
    )
    SELECT
        -- Distance to the *edge* of the target, not its centroid. The threat
        -- becomes relevant when it enters the zone, and a centroid measurement
        -- would systematically tell people they have more time than they do.
        -- NULL when no approach direction was extracted; 0 when the origin
        -- already overlaps the target.
        ST_Distance(o2.centroid::geography, core.g::geography) AS approach_meters,
        ST_AsEWKB(core.g) AS core_ewkb,
        buf.m AS buffer_meters,
        ST_AsEWKB(
            ST_UnaryUnion(
                ST_Collect(
                    ST_Buffer(core.g::geography, buf.m)::geometry,
                    COALESCE(corridor.g, ST_Buffer(core.g::geography, buf.m)::geometry)
                )
            )
        ) AS risk_ewkb
    FROM core, corridor, buf
    LEFT JOIN sectors o2 ON o2.id = CAST(:origin_id AS int)
    WHERE core.g IS NOT NULL
    """
)


#: Ceiling on a usable estimate, matching `ExtractedThreat.eta_minutes`'s own
#: bound. Past three hours the number is neither in contract nor actionable,
#: and in practice it means the origin matched the wrong sector.
_ETA_CEILING_MINUTES = 180


def estimate_eta_minutes(
    approach_meters: float | None, cruise_kmh: int, min_approach_meters: int
) -> int | None:
    """Time to reach the target zone at a nominal cruise speed.

    A fallback for the common case, not a model: monitoring channels state an
    arrival time in a small minority of messages, so without this `eta_minutes`
    is almost always null and the client has nothing to show. Returns ``None``
    whenever the inputs cannot support an honest number — no known approach
    direction, an origin already touching the target, or a result so large it
    can only be a mis-resolved sector.
    """
    if approach_meters is None or approach_meters < min_approach_meters:
        return None
    minutes = round(approach_meters / 1000.0 / cruise_kmh * 60.0)
    if minutes < 1 or minutes > _ETA_CEILING_MINUTES:
        return None
    return minutes


@dataclass(slots=True)
class RiskGeometry:
    """Geometry for one cluster, plus the distance the corridor implies."""

    core_ewkb: bytes
    risk_ewkb: bytes
    #: Metres from the approach sector to the near edge of the target, or None
    #: when no origin was given. Feeds `estimate_eta_minutes`.
    approach_meters: float | None
    #: The margin actually applied, after scaling by target size and clamping.
    #: Derived, not an input — logged so a surprising zone can be explained
    #: without re-running the query.
    buffer_meters: float


async def build_risk_geometry(
    session: AsyncSession,
    resolved: ResolvedSectors,
    buffer_ratio: float,
    buffer_floor_meters: int,
    buffer_ceiling_meters: int,
    point_radius_meters: int,
    origin_sector_id: int | None = None,
) -> RiskGeometry | None:
    """Build the geometry for a cluster.

    ``core`` is the union of the district polygons (and buffered coordinate
    points) — being inside it is a direct hit. ``risk`` adds the safety buffer
    and, when an approach direction is known, the corridor between origin and
    target. Returns ``None`` when the cluster could not be localised at all.

    The buffer is a fraction of the target's own width rather than a fixed
    distance; see ``Settings.buffer_ratio_shahed`` for why.
    """
    if not resolved.is_localisable:
        return None

    lats = [lat for lat, _ in resolved.points]
    lons = [lon for _, lon in resolved.points]

    row = (
        (
            await session.execute(
                _GEOMETRY_SQL,
                {
                    "sector_ids": resolved.sector_ids,
                    "lats": lats,
                    "lons": lons,
                    "point_radius": point_radius_meters,
                    "buffer_ratio": buffer_ratio,
                    "buffer_floor": buffer_floor_meters,
                    "buffer_ceiling": buffer_ceiling_meters,
                    "origin_id": origin_sector_id,
                },
            )
        )
        .mappings()
        .first()
    )
    if row is None or row["core_ewkb"] is None or row["risk_ewkb"] is None:
        return None
    approach = row["approach_meters"]
    return RiskGeometry(
        core_ewkb=bytes(row["core_ewkb"]),
        risk_ewkb=bytes(row["risk_ewkb"]),
        approach_meters=float(approach) if approach is not None else None,
        buffer_meters=float(row["buffer_meters"]),
    )
