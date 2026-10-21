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
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
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

_MATCH_SQL = text(
    f"""
    SELECT s.id, s.name, s.name_norm,
           ST_Y(s.centroid) AS lat, ST_X(s.centroid) AS lon,
           count(*) OVER (PARTITION BY (s.name_norm = :norm)) AS candidates
    FROM sectors s
    WHERE s.name_norm = :norm
       OR :norm = ANY(s.aliases)
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
_FUZZY_SQL = text(
    f"""
    SELECT s.id, s.name, s.name_norm, scored.score,
           ST_Y(s.centroid) AS lat, ST_X(s.centroid) AS lon,
           count(*) OVER (PARTITION BY scored.score) AS candidates
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
    ORDER BY scored.score DESC,
             {_ANCHOR_DISTANCE.format(t="s")} ASC,
             {_TYPE_RANK.format(t="s")} ASC,
             ST_Area(s.geom) DESC
    LIMIT 1
    """
)


def _anchor_wkt(points: list[tuple[float, float]]) -> str | None:
    """Everything in this report whose position is already certain."""
    if not points:
        return None
    coords = ", ".join(f"{lon} {lat}" for lat, lon in points)
    return f"MULTIPOINT({coords})"


async def _lookup(
    session: AsyncSession, norm: str, threshold: float, anchor: str | None
) -> Any | None:
    """One climb up the ladder: exact name, then alias, then trigram."""
    params = {"norm": norm, "anchor": anchor}
    row = (await session.execute(_MATCH_SQL, params)).mappings().first()
    if row is None:
        row = (
            (await session.execute(_FUZZY_SQL, {**params, "threshold": threshold}))
            .mappings()
            .first()
        )
    return row


async def resolve_sectors(
    session: AsyncSession, raw_names: list[str], threshold: float
) -> ResolvedSectors:
    """Map raw sector strings onto gazetteer rows and/or bare coordinates.

    Two passes. The first resolves every name on its own and records how many
    candidates tied for the win; the second re-resolves only the names that had
    a real choice, this time near the ones that did not. A single name in,
    a single name out — `resolve_origin` relies on that — but a report naming
    several places disambiguates itself.
    """
    resolved = ResolvedSectors()
    pending: list[tuple[str, str, Any | None]] = []

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

        pending.append((raw, norm, await _lookup(session, norm, threshold, None)))

    # A homonym is only as good as the company it keeps. Unambiguous matches —
    # and any raw coordinates — say where this report is.
    anchor = _anchor_wkt(
        [
            (row["lat"], row["lon"])
            for _, _, row in pending
            if row is not None and row["candidates"] == 1 and row["lat"] is not None
        ]
        + resolved.points
    )

    seen_ids: set[int] = set()
    for raw, norm, row in pending:
        if anchor is not None and row is not None and row["candidates"] > 1:
            # Keep the context-free match if the second pass finds nothing;
            # an anchor is a preference, never a filter.
            row = await _lookup(session, norm, threshold, anchor) or row

        if row is None:
            resolved.unmatched.append(raw)
            logger.info("sector_unmatched name=%r normalized=%r", raw, norm)
            continue

        if row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        resolved.sector_ids.append(row["id"])
        resolved.matched_names.append(row["name"])
        resolved.normalized_names.append(row["name_norm"])

    return resolved


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
