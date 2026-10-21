"""In-process spatial index — the whole point of the hot path.

Active threats in a country-scale deployment number in the hundreds, not the
millions. That fits comfortably in the address space of every API replica, so
``GET /threats/check`` does a Shapely ``STRtree`` probe against locally held
geometry and performs **zero network I/O**: no PostGIS round trip, no Redis
round trip, no serialization. A lookup is tens of microseconds.

Freshness is bounded by ``SNAPSHOT_POLL_SECONDS`` (default 2s). That is the
deliberate trade: two seconds of staleness against a TTL measured in tens of
minutes, in exchange for removing the database from the request path entirely.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy
import orjson
import shapely
from shapely import STRtree, wkb
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from app.cache import SnapshotThreat
from app.database import SectorRecord
from app.schemas import THREAT_PRIORITY, RiskLevel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ThreatMatch:
    threat_id: str
    threat_type: str
    risk_level: RiskLevel
    confidence_score: float
    sectors: list[str]
    #: Display spelling of `sectors`; see `database.ActiveThreatRecord`.
    target_sectors: list[str]
    eta_minutes: int | None
    origin_sector: str | None
    source_message_ids: list[int]
    updated_at: datetime
    #: Offset into the index that produced this match. Only meaningful against
    #: that same index — it is how a caller gets back to the matched geometry
    #: without a second point-in-polygon pass.
    position: int


class ThreatIndex:
    """Immutable snapshot of the active threat set, queryable by point."""

    __slots__ = ("_tree", "_threats", "_risk_geoms", "_core_geoms", "version", "built_at")

    def __init__(self, threats: list[SnapshotThreat], version: int | None = None) -> None:
        self._threats: list[SnapshotThreat] = []
        self._risk_geoms: list[BaseGeometry] = []
        self._core_geoms: list[BaseGeometry | None] = []

        for threat in threats:
            try:
                risk = wkb.loads(threat.geom_wkb)
            except Exception:  # malformed geometry must not poison the index
                logger.exception("threat_geometry_unreadable id=%s", threat.id)
                continue
            if risk.is_empty:
                continue
            core: BaseGeometry | None = None
            if threat.core_wkb:
                try:
                    core = wkb.loads(threat.core_wkb)
                except Exception:
                    logger.exception("threat_core_unreadable id=%s", threat.id)
            self._threats.append(threat)
            self._risk_geoms.append(risk)
            self._core_geoms.append(core)

        self._tree = STRtree(self._risk_geoms) if self._risk_geoms else None
        self.version = version
        self.built_at = datetime.now(UTC)

    def __len__(self) -> int:
        return len(self._threats)

    @property
    def threats(self) -> list[SnapshotThreat]:
        return self._threats

    def risk_geometry(self, position: int) -> BaseGeometry | None:
        """Exact risk geometry by position in `threats`. Read-only by contract."""
        return self._risk_geoms[position] if 0 <= position < len(self._risk_geoms) else None

    def core_geometry(self, position: int) -> BaseGeometry | None:
        return self._core_geoms[position] if 0 <= position < len(self._core_geoms) else None

    def lookup(
        self, latitude: float, longitude: float, high_confidence: float
    ) -> ThreatMatch | None:
        """Return the most severe threat covering the point, if any."""
        if self._tree is None:
            return None

        point = Point(longitude, latitude)  # PostGIS/GeoJSON order: x=lon, y=lat
        # `intersects` is the exact point-in-polygon test here, not a bbox hit.
        # Deliberately not "contains": STRtree applies the predicate as
        # `input.predicate(tree_geometry)`, so "contains" would ask whether the
        # *point* contains the polygon and silently match nothing. `intersects`
        # is symmetric, so argument order cannot bite, and it counts a point on
        # the boundary as inside — the safe direction for a shelter warning.
        candidates = self._tree.query(point, predicate="intersects")
        if len(candidates) == 0:
            return None

        now = datetime.now(UTC)
        best: ThreatMatch | None = None
        best_rank: tuple[int, int, float] = (-1, -1, -1.0)

        for idx in candidates:
            threat = self._threats[idx]
            if threat.expires_at <= now:
                continue  # snapshot is up to 2s stale; expiry is authoritative

            core = self._core_geoms[idx]
            # `intersects` rather than `contains` so the district boundary
            # itself counts as a direct hit.
            level: RiskLevel = (
                "high" if core is not None and core.intersects(point) else "elevated"
            )
            # A direct-hit polygon still only warrants "high" if we actually
            # believe the report.
            if level == "high" and threat.confidence_score < high_confidence:
                level = "elevated"

            rank = (
                2 if level == "high" else 1,
                THREAT_PRIORITY.get(threat.threat_type, 0),
                threat.confidence_score,
            )
            if rank > best_rank:
                best_rank = rank
                best = ThreatMatch(
                    threat_id=threat.id,
                    threat_type=threat.threat_type,
                    risk_level=level,
                    confidence_score=threat.confidence_score,
                    sectors=threat.sectors,
                    target_sectors=threat.target_sectors,
                    eta_minutes=threat.eta_minutes,
                    origin_sector=threat.origin_sector,
                    source_message_ids=threat.source_message_ids,
                    updated_at=threat.updated_at,
                    position=int(idx),
                )

        return best


class SectorIndex:
    """Static gazetteer index, used to name the district a user is standing in."""

    __slots__ = ("_tree", "_names", "_geoms", "_areas")

    def __init__(self, sectors: list[SectorRecord]) -> None:
        self._names: list[str] = []
        self._geoms: list[BaseGeometry] = []
        self._areas: list[float] = []
        for sector in sectors:
            try:
                geom = wkb.loads(sector.geom_wkb)
            except Exception:
                logger.exception("sector_geometry_unreadable id=%s", sector.id)
                continue
            if geom.is_empty:
                continue
            self._names.append(sector.name)
            self._geoms.append(geom)
            # Planar degree² area. Only ever compared between polygons that
            # overlap the same point, so the projection distortion cancels —
            # accurate enough to rank, and far cheaper than a geodesic area.
            self._areas.append(geom.area)
        self._tree = STRtree(self._geoms) if self._geoms else None

    def __len__(self) -> int:
        return len(self._names)

    def sector_at(self, latitude: float, longitude: float) -> str | None:
        """Name the *most specific* sector containing the point.

        Administrative boundaries nest — a coordinate in Obolon is also inside
        Kyiv oblast, and the tree returns both in arbitrary order. The smallest
        matching polygon is the one worth showing a user deciding whether to
        take shelter.
        """
        if self._tree is None:
            return None
        point = Point(longitude, latitude)
        hits = self._tree.query(point, predicate="intersects")  # see ThreatIndex.lookup
        if len(hits) == 0:
            return None
        return self._names[min(hits, key=lambda i: self._areas[i])]


def format_eta_window(
    eta_minutes: int | None, updated_at: datetime, now: datetime | None = None
) -> str | None:
    """Render a human-facing arrival window, decayed to the current time.

    An ETA recorded 4 minutes ago against a 10-minute estimate should read as
    ~6 minutes out, not 10. Returns ``None`` when no ETA was extracted.
    """
    if eta_minutes is None:
        return None

    now = now or datetime.now(UTC)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)

    elapsed_minutes = (now - updated_at).total_seconds() / 60.0
    remaining = eta_minutes - elapsed_minutes
    if remaining <= 0.5:
        return "now"

    # Round to whole minutes *before* widening, otherwise a freshly stored
    # 10-minute ETA reads as "7-13" because of sub-second elapsed time.
    centre = round(remaining)

    # Width grows with the estimate. A 10-minute ETA comes from a source that
    # watched the thing go past; a 45-minute one is almost always derived from
    # an oblast centroid, and the threat's real position inside an oblast is
    # unknown by tens of kilometres. Rendering both as ±2 minutes would be
    # false precision on the half that has none.
    #
    # Asymmetric on purpose, and the floors keep short ETAs reading exactly as
    # they did before: arriving early is the dangerous side to be wrong on.
    lower = max(2, round(centre * 0.15))
    upper = max(3, round(centre * 0.25))
    return f"{max(0, centre - lower)}–{centre + upper} min"


# ---------------------------------------------------------------------------
# Map overlay
# ---------------------------------------------------------------------------


#: How far the tolerance may escalate before the budget is given up on. Each
#: step doubles, so five steps is 32x the configured tolerance — enough to take
#: the largest shape the gazetteer can produce (a whole-oblast union, ~9,000
#: raw vertices) under 200, while still terminating on a pathological input
#: instead of thinning it away.
_SIMPLIFY_MAX_STEPS = 5


def _thin_to_budget(
    geom: BaseGeometry, tolerance: float, max_vertices: int
) -> BaseGeometry | None:
    """Douglas-Peucker (topology-preserving) with a vertex ceiling.

    A tolerance bounds a shape's *error*, not its *size*. Measured against the
    live gazetteer at 0.005 degrees: one Kyiv district thins to 21 vertices,
    the union of eight to 654, a whole oblast to 688. So the payload grows with
    how much of the country is under alert — worst exactly on the night a
    client can least afford a slow poll.

    Doubling the tolerance until the shape fits trades away resolution the map
    cannot show anyway: an oblast needs ~0.04 degrees, which still lands a
    vertex every ~3 km, under a pixel at the zoom where a whole oblast is on
    screen. A single district never escalates at all.

    Each pass thins the *previous* result rather than restarting from the
    original. Measured over a 12-oblast worst case (117,616 raw vertices), that
    is 82 ms against 206 ms for re-simplifying the source every time, because
    the escalating passes run over hundreds of vertices instead of thousands.
    The shapes come out the same: identical vertex counts, and a symmetric
    difference of 0.000% of area at the median and 0.094% at the worst.

    Returns the last non-empty result — a shape that is slightly too detailed
    beats one thinned out of existence — or ``None`` if even the first pass is
    empty. ``max_vertices <= 0`` disables the ceiling.
    """
    best = geom.simplify(tolerance, preserve_topology=True)
    if best.is_empty:
        return None
    if max_vertices <= 0:
        return best

    for step in range(1, _SIMPLIFY_MAX_STEPS + 1):
        if shapely.get_num_coordinates(best) <= max_vertices:
            return best
        thinned = best.simplify(tolerance * (2**step), preserve_topology=True)
        if thinned.is_empty:
            break  # keep the previous, finer-but-real shape
        best = thinned

    remaining = shapely.get_num_coordinates(best)
    if remaining > max_vertices:
        logger.warning(
            "display_geometry_over_budget vertices=%d budget=%d — sending anyway",
            remaining,
            max_vertices,
        )
    return best


def _display_geometry(
    geom: BaseGeometry | None, simplify_degrees: float, precision: int, max_vertices: int
) -> BaseGeometry | None:
    """Thin a lookup geometry down to something worth sending over the wire.

    Two Features per threat cost ~12 KB at full precision and ~6 KB once
    simplified; on a heavy night that is the difference between a 200 KB poll
    and a 40 KB one. The tolerance is an order of magnitude below the buffer
    radii it is applied to, so the shapes stay visually identical, and the
    vertex budget caps what a many-district union can cost.

    Never used for lookups. `ThreatIndex` keeps the exact geometry, and the
    answer to "am I in a threat zone" is computed from that.
    """
    if geom is None:
        return None
    try:
        thinned = _thin_to_budget(geom, simplify_degrees, max_vertices)
        if thinned is None:
            return None
        # Rounding after simplification cannot disturb topology: the surviving
        # vertices are two orders of magnitude further apart than the grid.
        rounded = shapely.transform(thinned, lambda c: numpy.round(c, precision))
        return None if rounded.is_empty else rounded
    except Exception:
        logger.exception("display_geometry_failed — omitting this shape")
        return None


def threat_geometry_geojson(
    index: ThreatIndex,
    position: int,
    simplify_degrees: float,
    precision: int,
    max_vertices: int,
) -> dict[str, object] | None:
    """The display shape for one threat, as a bare GeoJSON geometry object.

    Same thinning as the overlay, one threat instead of all of them: what
    ``/threats/check`` attaches so a caller that never fetches the full overlay
    — a widget, a Live Activity — can still draw the zone it was told it is in.
    Returns the ``risk`` shape, which is the one ``/check`` answers from.
    """
    geom = index.risk_geometry(position)
    if geom is None:
        return None
    display = _display_geometry(geom, simplify_degrees, precision, max_vertices)
    if display is None:
        return None
    return orjson.loads(shapely.to_geojson(display))


def _union_features(
    index: ThreatIndex,
    high_confidence: float,
    simplify_degrees: float,
    precision: int,
    max_vertices: int,
    now: datetime,
) -> list[dict[str, object]]:
    """The layer a client fills: one disjoint pair of shapes per threat type.

    Two shapes, split where the *status line* splits rather than where the
    geometry does:

    * ``core_union`` — reported districts the server would actually rate
      ``high``. A direct hit you are inside of.
    * ``risk_union`` — every buffer and corridor, **plus** the reported
      districts whose confidence falls under ``high_confidence``. Those are
      cores geometrically, but `ThreatIndex.lookup` will only ever call them
      ``elevated``, so painting them as direct hits is what puts a dense
      polygon under a user whose own status reads `elevated`.

    ``risk_union`` has ``core_union`` subtracted out, so within one threat type
    no pixel is covered twice and the saturation on screen is a constant that
    means something. Two *different* types can still overlap — a missile wash
    under a shahed wash is a real statement about a real situation, and the
    palette is per type, so merging those would be a lie in the other
    direction.

    Both are ranked by the same threat-type accent a client already draws, and
    both are thinned exactly like the per-threat shapes: a wash whose edge does
    not sit on the outline beside it reads as a rendering bug.
    """
    buckets: dict[str, dict[str, list]] = {}

    for position, threat in enumerate(index.threats):
        if threat.expires_at <= now:
            continue
        bucket = buckets.setdefault(
            threat.threat_type, {"hit": [], "warn": [], "threats": []}
        )
        core = index.core_geometry(position)
        if core is not None:
            bucket["hit" if threat.confidence_score >= high_confidence else "warn"].append(core)
        risk = index.risk_geometry(position)
        if risk is not None:
            bucket["warn"].append(risk)
        bucket["threats"].append(threat)

    features: list[dict[str, object]] = []
    for threat_type, bucket in buckets.items():
        threats = bucket["threats"]
        try:
            hit = shapely.union_all(bucket["hit"]) if bucket["hit"] else None
            warn = shapely.union_all(bucket["warn"]) if bucket["warn"] else None
            # Thin first, subtract second. Thinning moves a boundary by up to
            # the tolerance, so subtracting the exact shapes and thinning the
            # result leaves a hairline of overlap along every shared edge —
            # the one place a doubled alpha is most visible. Differencing the
            # shapes that actually go on the wire cannot.
            hit = _display_geometry(hit, simplify_degrees, precision, max_vertices)
            warn = _display_geometry(warn, simplify_degrees, precision, max_vertices)
            if warn is not None and hit is not None:
                warn = warn.difference(hit)
        except Exception:
            # A wash is a nicety; the per-threat Features above are the payload
            # that matters, and they are already in `features`.
            logger.exception("union_geometry_failed type=%s — omitting the wash", threat_type)
            continue

        properties = {
            "id": f"union-{threat_type}",
            "threat_type": threat_type,
            "count": sum(t.count for t in threats),
            # The strongest report in the union sets its weight. A wash is one
            # shape and can only carry one confidence; the alternative — the
            # weakest — would fade a corroborated sighting because something
            # marginal happened to touch it.
            "confidence_score": max(t.confidence_score for t in threats),
            "sectors": sorted({sector for t in threats for sector in t.sectors}),
            "target_sectors": sorted({s for t in threats for s in t.target_sectors}),
            # Deliberately absent: a merged shape has no one arrival time, and
            # a client reading a window off it would be quoting a threat the
            # user may be nowhere near.
            "eta_minutes": None,
            # The oldest member: the wash has been up since its first threat was.
            "first_seen_at": min(t.first_seen_at for t in threats).isoformat(),
            "updated_at": max(t.updated_at for t in threats).isoformat(),
            # The latest, so the wash cannot blink out from under threats that
            # are still live. It can outlast its earliest member by up to one
            # janitor interval; the per-threat outlines expire on their own
            # clock, so what lingers is a slightly over-large wash rather than
            # a zone claiming to still be there.
            "expires_at": max(t.expires_at for t in threats).isoformat(),
        }

        for kind, geom in (("risk_union", warn), ("core_union", hit)):
            if geom is None or geom.is_empty:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": orjson.Fragment(shapely.to_geojson(geom).encode()),
                    "properties": {**properties, "kind": kind},
                }
            )

    return features


def threats_geojson(
    index: ThreatIndex,
    simplify_degrees: float,
    precision: int,
    max_vertices: int,
    high_confidence: float = 0.65,
    now: datetime | None = None,
) -> bytes:
    """Serialise the whole active threat set as a GeoJSON FeatureCollection.

    Two Features per threat, distinguished by ``kind``:

    * ``core`` — the districts actually named in the reports. A direct hit.
    * ``risk`` — core plus the safety buffer and, when an approach direction
      was known, the corridor. This is the shape ``/threats/check`` answers
      from.

    Drawing them as one undifferentiated blob would overstate what is known:
    the buffer is precaution, not observation, and on these radii it dwarfs the
    core.

    Those two are for *asking* — point-in-polygon, captions, arrival windows —
    and they overlap by construction: `risk` contains its own `core`, and two
    reports naming neighbouring districts overlap again. Painting them directly
    composites alpha, so the saturation on screen ends up encoding how many
    reports piled up rather than how dangerous the place is.

    So the payload also carries a second, *disjoint* pair per threat type —
    ``core_union`` and ``risk_union`` — which is what a client should actually
    fill. See `_union_features`.

    No ETA window is baked in — it decays by the second, and this payload is
    cached per snapshot version. `eta_minutes` and `updated_at` travel raw so
    the client can run `format_eta_window` itself and let the countdown tick
    between polls.
    """
    now = now or datetime.now(UTC)
    features: list[dict[str, object]] = []

    for position, threat in enumerate(index.threats):
        if threat.expires_at <= now:
            continue
        shapes: list[tuple[str, BaseGeometry | None]] = [
            ("core", index.core_geometry(position)),
            ("risk", index.risk_geometry(position)),
        ]
        for kind, geom in shapes:
            if geom is None:
                continue
            display = _display_geometry(geom, simplify_degrees, precision, max_vertices)
            if display is None:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": orjson.Fragment(shapely.to_geojson(display).encode()),
                    "properties": {
                        "id": threat.id,
                        "kind": kind,
                        "threat_type": threat.threat_type,
                        "count": threat.count,
                        "confidence_score": threat.confidence_score,
                        "sectors": threat.sectors,
                        "target_sectors": threat.target_sectors,
                        "eta_minutes": threat.eta_minutes,
                        # Whether standing in this shape is a direct hit, on
                        # the same threshold `lookup` applies before it will
                        # say `high`. Always false for `risk`, which is
                        # precaution by definition. A client drawing every
                        # core alike shows a dense "direct hit" polygon around
                        # a user whose own status line reads `elevated`.
                        "direct_hit": kind == "core"
                        and threat.confidence_score >= high_confidence,
                        # When the threat was first reported; `updated_at`
                        # moves on every re-report, so it cannot anchor a
                        # "time since alert" clock.
                        "first_seen_at": threat.first_seen_at.isoformat(),
                        "updated_at": threat.updated_at.isoformat(),
                        "expires_at": threat.expires_at.isoformat(),
                    },
                }
            )

    features.extend(
        _union_features(index, high_confidence, simplify_degrees, precision, max_vertices, now)
    )

    return orjson.dumps(
        {
            "type": "FeatureCollection",
            "generated_at": now.isoformat(),
            "version": index.version,
            "features": features,
        }
    )
