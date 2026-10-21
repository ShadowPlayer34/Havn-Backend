"""FastAPI surface.

The contract with the iOS client is one latency-critical endpoint. Everything
here is arranged around keeping that endpoint free of I/O: the spatial index is
refreshed by a background task, and the request handler only reads memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app import cache, database
from app.config import Settings, get_settings
from app.schemas import (
    ActiveThreat,
    NearestSheltersResponse,
    ShelterOut,
    ShelterReportAck,
    ShelterReportRequest,
    ThreatCheckResponse,
)
from app.spatial import (
    SectorIndex,
    ThreatIndex,
    format_eta_window,
    threat_geometry_geojson,
    threats_geojson,
)

logger = logging.getLogger(__name__)


class AppState:
    """Mutable process-local state, swapped atomically by the refresher."""

    def __init__(self) -> None:
        self.threat_index: ThreatIndex = ThreatIndex([])
        self.sector_index: SectorIndex = SectorIndex([])
        self.last_refresh_at: datetime | None = None
        self.snapshot_version: int | None = None
        self._zones_cache: tuple[ThreatIndex, bytes] | None = None
        self._check_geometry_cache: tuple[ThreatIndex, dict[int, dict | None]] | None = None

    def zones_geojson(self, settings: Settings) -> bytes:
        """Serialised map overlay for the current index, built at most once.

        The payload is identical for every client, so building it per request
        would be pure waste. Keyed on the index *object*: the refresher swaps in
        a new instance whenever the snapshot version moves, which invalidates
        this by construction and cannot go stale in the other direction.

        Serialisation is synchronous and holds the event loop, so concurrent
        first-requesters cannot race into building it twice.
        """
        index = self.threat_index
        cached = self._zones_cache
        if cached is not None and cached[0] is index:
            return cached[1]
        payload = threats_geojson(
            index,
            settings.zones_simplify_degrees,
            settings.zones_precision,
            settings.zones_max_vertices,
            settings.high_confidence,
        )
        self._zones_cache = (index, payload)
        logger.info("zones_serialised bytes=%d version=%s", len(payload), index.version)
        return payload

    def check_geometry(
        self, index: ThreatIndex, position: int, settings: Settings
    ) -> dict | None:
        """Display shape for one matched threat, thinned at most once per index.

        Same invalidation as `zones_geojson`: keyed on the index object, which
        the refresher replaces wholesale. Populated per threat rather than up
        front — a night with fifty clusters is one where nobody is asking about
        the forty-nine they are not standing in.

        The index is a parameter rather than a read of `self.threat_index`
        because `position` only means anything against the index that produced
        the match. Today no handler awaits between the two, so they cannot
        differ; passing it explicitly means that stays true when one does.

        The returned dict is shared between requests and must not be mutated.
        """
        cached = self._check_geometry_cache
        if cached is None or cached[0] is not index:
            cached = (index, {})
            self._check_geometry_cache = cached
        shapes = cached[1]
        if position not in shapes:
            shapes[position] = threat_geometry_geojson(
                index,
                position,
                settings.zones_simplify_degrees,
                settings.zones_precision,
                settings.zones_max_vertices,
            )
        return shapes[position]


state = AppState()


async def _load_from_database(
    settings: Settings, sessionmaker: async_sessionmaker[AsyncSession]
) -> ThreatIndex:
    """Cold-start / degraded path: build the index straight from PostGIS."""
    async with database.session_scope(sessionmaker) as session:
        records = await database.load_active_threats(session, settings.min_confidence)
    snapshot = [
        cache.SnapshotThreat(
            id=r.id,
            threat_type=r.threat_type,
            count=r.count,
            confidence_score=r.confidence_score,
            sectors=r.sectors,
            target_sectors=r.target_sectors,
            eta_minutes=r.eta_minutes,
            origin_sector=r.origin_sector,
            first_seen_at=r.first_seen_at,
            updated_at=r.updated_at,
            expires_at=r.expires_at,
            geom_wkb=r.geom_wkb,
            core_wkb=r.core_wkb,
            source_message_ids=r.source_message_ids,
        )
        for r in records
    ]
    return ThreatIndex(snapshot)


async def _try_load_sectors(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> SectorIndex | None:
    """Load the gazetteer, tolerating a schema that does not exist yet.

    On a cold `docker compose up` the API can win the race against the worker's
    schema bootstrap. That must not be fatal — the refresher retries.
    """
    try:
        async with database.session_scope(sessionmaker) as session:
            return SectorIndex(await database.load_sectors(session))
    except Exception:
        logger.warning("sector_load_failed — will retry from the refresher", exc_info=True)
        return None


async def _refresh_loop(app: FastAPI) -> None:
    """Poll the snapshot version; rebuild the index only when it moves."""
    settings: Settings = app.state.settings
    redis = app.state.redis
    sessionmaker = app.state.sessionmaker
    stale_after = max(30.0, settings.snapshot_poll_seconds * 10)

    while True:
        try:
            if len(state.sector_index) == 0:
                sectors = await _try_load_sectors(sessionmaker)
                if sectors is not None and len(sectors):
                    state.sector_index = sectors
                    logger.info("sector_index_loaded sectors=%d", len(sectors))

            version = await cache.get_snapshot_version(redis)
            if version is not None and version != state.snapshot_version:
                threats = await cache.get_snapshot(redis)
                if threats is not None:
                    state.threat_index = ThreatIndex(threats, version=version)
                    state.snapshot_version = version
                    state.last_refresh_at = datetime.now(UTC)
                    logger.info(
                        "index_refreshed source=redis threats=%d version=%d",
                        len(state.threat_index),
                        version,
                    )
            elif version is None:
                # No snapshot at all (worker down, Redis flushed). Rebuild from
                # the source of truth rather than serving an empty index.
                age = (
                    (datetime.now(UTC) - state.last_refresh_at).total_seconds()
                    if state.last_refresh_at
                    else float("inf")
                )
                if age > stale_after:
                    state.threat_index = await _load_from_database(settings, sessionmaker)
                    state.snapshot_version = None
                    state.last_refresh_at = datetime.now(UTC)
                    logger.warning(
                        "index_refreshed source=postgis threats=%d (no redis snapshot)",
                        len(state.threat_index),
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("index_refresh_failed — serving previous index")

        await asyncio.sleep(settings.snapshot_poll_seconds)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    engine = database.create_engine(settings)
    sessionmaker = database.create_sessionmaker(engine)
    redis = cache.create_redis(settings.redis_url)

    app.state.settings = settings
    app.state.engine = engine
    app.state.sessionmaker = sessionmaker
    app.state.redis = redis

    # The gazetteer is static reference data — load once, never on request.
    sectors = await _try_load_sectors(sessionmaker)
    if sectors is not None:
        state.sector_index = sectors
    logger.info("sector_index_loaded sectors=%d", len(state.sector_index))

    try:
        threats = await cache.get_snapshot(redis)
        if threats is not None:
            # The version has to go *into* the index, not just alongside it:
            # the overlay ETag is keyed on it, and the refresher will not
            # rebuild until the version moves — so a cold start that skipped it
            # would serve un-cacheable zones until the next window landed.
            version = await cache.get_snapshot_version(redis)
            state.threat_index = ThreatIndex(threats, version=version)
            state.snapshot_version = version
        else:
            state.threat_index = await _load_from_database(settings, sessionmaker)
        state.last_refresh_at = datetime.now(UTC)
    except Exception:
        # Serve "no known threats" rather than refusing to boot; the refresher
        # will populate the index as soon as the dependency recovers.
        logger.warning("initial_threat_load_failed — starting empty", exc_info=True)
    logger.info("threat_index_loaded threats=%d", len(state.threat_index))

    refresher = asyncio.create_task(_refresh_loop(app), name="snapshot-refresher")
    try:
        yield
    finally:
        refresher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await refresher
        await redis.aclose()
        await engine.dispose()


app = FastAPI(
    title="AlertRadar API",
    version="1.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

# Coordinate lists are extremely repetitive and compress ~5x. The threshold
# keeps it away from the hot path, whose responses are a couple of hundred
# bytes and would only pay the CPU without shrinking a packet.
app.add_middleware(GZipMiddleware, minimum_size=1024)


def get_app_settings() -> Settings:
    return get_settings()


async def get_session() -> AsyncIterator[AsyncSession]:
    """Request-scoped session for the endpoints that actually touch PostGIS."""
    async with database.session_scope(app.state.sessionmaker) as session:
        yield session


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, object]:
    age = (
        (datetime.now(UTC) - state.last_refresh_at).total_seconds()
        if state.last_refresh_at
        else None
    )
    return {
        "status": "ok",
        "active_threats": len(state.threat_index),
        "sectors": len(state.sector_index),
        "snapshot_version": state.snapshot_version,
        "index_age_seconds": age,
    }


@app.get(
    "/api/v1/threats/check",
    response_model=ThreatCheckResponse,
    summary="Is this coordinate inside an active threat zone?",
)
async def check_threats(
    response: Response,
    settings: Annotated[Settings, Depends(get_app_settings)],
    latitude: Annotated[float, Query(ge=-90.0, le=90.0)],
    longitude: Annotated[float, Query(ge=-180.0, le=180.0)],
    include_geometry: Annotated[
        bool,
        Query(
            description="Attach the matched `risk` polygon as GeoJSON. Off by "
            "default: this response is uncacheable and polled continuously, so "
            "a client that draws a map should take the ETag-cached "
            "/threats/zones overlay instead of paying for geometry every 30 s."
        ),
    ] = False,
) -> ThreatCheckResponse:
    # Never let a CDN or the URL loader cache a safety answer.
    response.headers["Cache-Control"] = "no-store"

    index = state.threat_index
    match = index.lookup(latitude, longitude, settings.high_confidence)
    if match is None:
        return ThreatCheckResponse(in_risk_zone=False, risk_level="none")

    # Prefer the district the user is actually standing in over the threat's
    # (possibly multi-district) sector list.
    sector_name = state.sector_index.sector_at(latitude, longitude)
    if sector_name is None and match.target_sectors:
        # `target_sectors`, not `sectors`: the latter holds folded match keys
        # ("дарницький") and this value goes on screen.
        sector_name = match.target_sectors[0]

    geometry = (
        state.check_geometry(index, match.position, settings) if include_geometry else None
    )

    return ThreatCheckResponse(
        in_risk_zone=True,
        risk_level=match.risk_level,
        threat_type=match.threat_type,
        eta_window=format_eta_window(match.eta_minutes, match.updated_at),
        # The stored estimate, undecayed — `eta_window` is the same number
        # rendered against `now`. A client that wants a countdown to tick
        # between 30s polls needs the raw one; see ThreatCheckResponse.
        eta_minutes=match.eta_minutes,
        sector_name=sector_name,
        target_sectors=match.target_sectors,
        origin_sector=match.origin_sector,
        source_refs=match.source_message_ids,
        geometry=geometry,
    )


@app.get(
    "/api/v1/threats/active",
    response_model=list[ActiveThreat],
    summary="All active threat clusters (map overlay)",
)
async def active_threats() -> list[ActiveThreat]:
    """Metadata only — geometry is intentionally excluded to keep it light.

    Not part of the original spec; added because the iOS radar map needs a
    non-point-query view of the current state.
    """
    now = datetime.now(UTC)
    return [
        ActiveThreat(
            id=t.id,
            threat_type=t.threat_type,
            count=t.count,
            confidence_score=t.confidence_score,
            sectors=t.sectors,
            target_sectors=t.target_sectors,
            expires_at=t.expires_at,
            updated_at=t.updated_at,
            eta_minutes=t.eta_minutes,
        )
        for t in state.threat_index.threats
        if t.expires_at > now
    ]


@app.get(
    "/api/v1/threats/zones",
    summary="All active threat zones as GeoJSON (map overlay)",
    response_class=Response,
    responses={200: {"content": {"application/geo+json": {}}}, 304: {}},
)
async def threat_zones(
    request: Request,
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> Response:
    """The country-wide view, as opposed to `/check`'s answer about one point.

    Two Features per threat — `core` (districts actually reported) and `risk`
    (core plus buffer and corridor) — so a client can draw observation and
    precaution differently. They should be: on these buffer radii the risk
    shape is several times the core, and rendering them alike would paint most
    of the country red on any active night.

    Needs no coordinates. A client can hold the overlay and answer "am I in a
    zone" locally without ever telling the server where its user is.

    Cached per snapshot version and revalidated with an ETag, so the steady
    state is a 304 with no body. A zone that expires between version bumps can
    linger in the payload for up to one janitor interval; every Feature carries
    `expires_at` so clients can drop those without waiting.
    """
    index = state.threat_index
    etag = f'W/"zones-{index.version}"' if index.version is not None else None

    # `no-cache` rather than `no-store`: revalidate every time, but let the 304
    # path work. `no-store` would forbid holding the body at all and turn every
    # poll back into a full download.
    headers = {"Cache-Control": "no-cache"}
    if etag is not None:
        headers["ETag"] = etag
        candidates = {
            part.strip() for part in request.headers.get("if-none-match", "").split(",")
        }
        if etag in candidates or "*" in candidates:
            return Response(status_code=304, headers=headers)

    return Response(
        content=state.zones_geojson(settings),
        media_type="application/geo+json",
        headers=headers,
    )


@app.get(
    "/api/v1/shelters/nearest",
    response_model=NearestSheltersResponse,
    summary="Closest shelters to a coordinate, nearest first",
)
async def nearest_shelters(
    session: Annotated[AsyncSession, Depends(get_session)],
    latitude: Annotated[float, Query(ge=-90.0, le=90.0)],
    longitude: Annotated[float, Query(ge=-180.0, le=180.0)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> NearestSheltersResponse:
    """Served straight from PostGIS rather than the in-process index.

    The shelter table is static and the GiST KNN is sub-millisecond, so there is
    nothing to gain from another cached snapshot to keep coherent.
    """
    rows = await database.find_nearest_shelters(session, latitude, longitude, limit)
    return NearestSheltersResponse(
        shelters=[
            ShelterOut(
                id=r.id,
                name=r.name,
                shelter_type=r.shelter_type,
                source=r.source,
                address=r.address,
                district=r.district,
                kind=r.kind,
                building_type=r.building_type,
                working_time=r.working_time,
                phone=r.phone,
                accessible=r.accessible,
                notes=r.notes,
                reports=r.reports,
                latitude=r.latitude,
                longitude=r.longitude,
                distance_meters=round(r.distance_meters, 1),
            )
            for r in rows
        ]
    )


@app.post(
    "/api/v1/shelters/{shelter_id}/report",
    response_model=ShelterReportAck,
    status_code=202,
    summary="Tell us a shelter entry is wrong",
)
async def report_shelter(
    shelter_id: str,
    payload: ShelterReportRequest,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> ShelterReportAck:
    """Accepted, not acted on automatically.

    People on the ground are the only source that can correct a register at scale —
    an oblast pin can sit kilometres from its own address and nothing in the data
    says so. But reports cannot be allowed to remove a shelter by themselves, or
    anyone could clear the map for a district. Flagged entries sink in the ranking
    and surface for review; a human decides.
    """
    try:
        count = await database.record_shelter_report(
            session,
            shelter_id=shelter_id,
            reason=payload.reason,
            note=payload.note,
            client_id=payload.client_id,
        )
    except LookupError:
        raise HTTPException(status_code=404, detail="No such shelter") from None
    except database.ReportRateLimited:
        logger.warning("shelter_report_rate_limited client=%s", payload.client_id[:8])
        raise HTTPException(
            status_code=429,
            detail="Too many shelters reported from this device in the last hour.",
        ) from None

    logger.info("shelter_reported id=%s reason=%s total=%d", shelter_id, payload.reason, count)
    return ShelterReportAck(shelter_id=shelter_id, reports=count)
