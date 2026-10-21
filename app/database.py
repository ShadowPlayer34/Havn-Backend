"""Spatial state store: SQLAlchemy 2.0 models, upserts, TTL cleanup, queries.

Design notes
------------
* PostGIS is the **source of truth**. Redis holds a derived snapshot that the
  API reads; it is always rebuildable from here.
* Expiry is expressed twice on purpose: every read filters ``expires_at >
  now()`` (correct the instant a threat lapses, no janitor race), and a
  periodic janitor archives + deletes the rows (keeps the working set small so
  the GiST index stays hot).
* Geometry values cross the Python boundary as EWKB ``bytes``. GeoAlchemy2 is
  used for DDL and column typing; reads go through explicit ``ST_AsBinary`` /
  ``ST_AsEWKB`` so nothing depends on driver-level geometry codecs.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import Settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Sector(Base):
    """Administrative-district gazetteer. Static reference data."""

    __tablename__ = "sectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    #: Output of ``schemas.normalize_sector`` — the exact-match lookup key.
    name_norm: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    aliases: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    oblast: Mapped[str | None] = mapped_column(String(120), nullable=True)
    geom: Mapped[bytes] = mapped_column(
        Geometry(geometry_type="MULTIPOLYGON", srid=4326, spatial_index=True),
        nullable=False,
    )
    centroid: Mapped[bytes | None] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=False),
        nullable=True,
    )


class Shelter(Base):
    """Shelter points of interest, loaded from an OSM extract. Static reference data.

    ``id`` carries the OSM identity ("n123" for a node, "a456" for an area) rather
    than a serial, so re-importing an updated extract updates rows in place instead
    of duplicating them.

    Ways and relations are reduced to their centroid at load time: the KNN operator
    ``<->`` is cheapest against points, and a shelter's footprint is irrelevant when
    the question is "how far do I have to run".
    """

    __tablename__ = "shelters"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(
        String(200), nullable=False, server_default=text("'Укриття'")
    )
    address: Mapped[str | None] = mapped_column(String(300), nullable=True)
    #: Normalised category: "bomb_shelter", "dual_purpose", "simple_shelter",
    #: "bunker"… Kept coarse so the client can rank, with the registry's own
    #: wording preserved in ``kind``.
    shelter_type: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)

    #: Where the row came from — "kmda" (official register) or "osm". The two
    #: differ enormously in trustworthiness and the client is entitled to know.
    source: Mapped[str] = mapped_column(String(16), nullable=False, server_default=text("'osm'"))
    district: Mapped[str | None] = mapped_column(String(120), nullable=True)
    #: The registry's own words: "Підвал", "Підвал з пандусом ОЗ", …
    kind: Mapped[str | None] = mapped_column(String(160), nullable=True)
    #: What the shelter is inside: "Школа", "Житловий будинок", …
    building_type: Mapped[str | None] = mapped_column(String(160), nullable=True)
    #: "Цілодобово" (24/7) or a restriction like "only during the school day" —
    #: the single most decision-relevant field at 3am.
    working_time: Mapped[str | None] = mapped_column(String(300), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(120), nullable=True)
    owner: Mapped[str | None] = mapped_column(String(300), nullable=True)
    #: True when the register records a ramp / accessible entrance.
    accessible: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    geom: Mapped[bytes] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=True),
        nullable=False,
    )


class ShelterReport(Base):
    """A user telling us a shelter entry is wrong.

    Kept deliberately free of anything identifying: a reason, an optional note, and a
    client-generated UUID whose only job is to stop one device inflating a count. No
    reporter position is stored — knowing where someone was standing during an air
    raid is not worth the risk of holding it.
    """

    __tablename__ = "shelter_reports"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, server_default=func.gen_random_uuid()
    )
    shelter_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    #: wrong_location | locked | does_not_exist | wrong_hours | other
    reason: Mapped[str] = mapped_column(String(24), nullable=False)
    note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    client_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # One device, one voice per problem per shelter.
        Index("ux_shelter_reports_dedup", "shelter_id", "client_id", "reason", unique=True),
    )


class ActiveThreatRow(Base):
    """A live threat cluster with its risk geometry."""

    __tablename__ = "active_threats"

    id: Mapped[str] = mapped_column(
        String(64), primary_key=True, server_default=func.gen_random_uuid()
    )
    dedup_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    threat_type: Mapped[str] = mapped_column(String(16), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sectors: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    eta_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Gazetteer name of the district the threat is approaching FROM — the
    #: origin the corridor in ``geom`` was drawn from. Null when the extraction
    #: named no origin, or named one no sector matched: either way no corridor
    #: exists, so there is nothing here to attribute one to. The model's own
    #: wording survives in ``raw`` regardless.
    origin_sector: Mapped[str | None] = mapped_column(String(160), nullable=True)

    #: The district polygons themselves — being inside means "high".
    core_geom: Mapped[bytes | None] = mapped_column(
        Geometry(geometry_type="GEOMETRY", srid=4326, spatial_index=False),
        nullable=True,
    )
    #: Core + directional corridor + safety buffer — inside means "elevated".
    geom: Mapped[bytes] = mapped_column(
        Geometry(geometry_type="GEOMETRY", srid=4326, spatial_index=True),
        nullable=False,
    )

    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    source_channel: Mapped[str | None] = mapped_column(String(160), nullable=True)
    source_message_ids: Mapped[list[int]] = mapped_column(
        ARRAY(BigInteger), nullable=False, server_default=text("'{}'::bigint[]")
    )
    raw: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))

    __table_args__ = (
        Index("ix_active_threats_live", "expires_at", "threat_type"),
    )


class ThreatHistoryRow(Base):
    """Append-only archive of expired clusters (audit / model evaluation)."""

    __tablename__ = "threat_history"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dedup_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    threat_type: Mapped[str] = mapped_column(String(16), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sectors: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'::text[]")
    )
    geom: Mapped[bytes | None] = mapped_column(
        Geometry(geometry_type="GEOMETRY", srid=4326, spatial_index=False),
        nullable=True,
    )
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw: Mapped[dict] = mapped_column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))


# ---------------------------------------------------------------------------
# Engine / session plumbing
# ---------------------------------------------------------------------------


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        # Server-side statement cache and pgbouncer do not mix; disable if you
        # put a transaction pooler in front of this.
        echo=False,
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ThreatUpsert:
    """Everything needed to persist one resolved cluster."""

    dedup_key: str
    threat_type: str
    count: int
    confidence_score: float
    sectors: list[str]
    ttl_minutes: int
    eta_minutes: int | None
    origin_sector: str | None
    core_ewkb: bytes
    risk_ewkb: bytes
    source_channel: str | None
    source_message_ids: list[int]
    raw: dict


_UPSERT_SQL = text(
    """
    WITH existing AS (
        SELECT id
        FROM active_threats
        WHERE dedup_key = :dedup_key AND expires_at > now()
        ORDER BY expires_at DESC
        LIMIT 1
        FOR UPDATE
    ),
    updated AS (
        UPDATE active_threats a
        SET
            -- "max count in window" also holds across windows: a cluster never
            -- shrinks, it only expires.
            count              = GREATEST(a.count, :count),
            confidence_score   = GREATEST(a.confidence_score, :confidence),
            eta_minutes        = COALESCE(:eta_minutes, a.eta_minutes),
            -- Same rule as the ETA: a later window that did not mention an
            -- approach direction is silence, not a retraction. The corridor
            -- geometry below is rebuilt from this window alone, so a re-report
            -- without an origin does drop the corridor — but the name we show
            -- keeps pointing at the direction last actually reported.
            origin_sector      = COALESCE(:origin_sector, a.origin_sector),
            sectors            = ARRAY(
                                     SELECT DISTINCT unnest(a.sectors || CAST(:sectors AS text[]))
                                 ),
            core_geom          = ST_GeomFromEWKB(:core_ewkb),
            geom               = ST_GeomFromEWKB(:risk_ewkb),
            updated_at         = now(),
            -- Re-reporting refreshes the countdown rather than extending it
            -- forever; the window never shortens.
            expires_at         = GREATEST(
                                     a.expires_at,
                                     now() + make_interval(mins => :ttl_minutes)
                                 ),
            source_message_ids = ARRAY(
                                     SELECT DISTINCT unnest(
                                         a.source_message_ids || CAST(:message_ids AS bigint[])
                                     )
                                 ),
            raw                = CAST(:raw AS jsonb)
        FROM existing e
        WHERE a.id = e.id
        RETURNING a.id
    ),
    inserted AS (
        INSERT INTO active_threats (
            id, dedup_key, threat_type, count, confidence_score, sectors,
            eta_minutes, origin_sector, core_geom, geom, first_seen_at,
            updated_at, expires_at, source_channel, source_message_ids, raw
        )
        SELECT
            gen_random_uuid()::text, :dedup_key, :threat_type, :count, :confidence,
            CAST(:sectors AS text[]), :eta_minutes, :origin_sector,
            ST_GeomFromEWKB(:core_ewkb), ST_GeomFromEWKB(:risk_ewkb),
            now(), now(), now() + make_interval(mins => :ttl_minutes),
            :source_channel, CAST(:message_ids AS bigint[]), CAST(:raw AS jsonb)
        WHERE NOT EXISTS (SELECT 1 FROM existing)
        RETURNING id
    )
    SELECT id FROM updated UNION ALL SELECT id FROM inserted
    """
)


async def upsert_threat(session: AsyncSession, payload: ThreatUpsert) -> str:
    """Insert or merge a cluster, returning its id.

    Concurrency: an advisory transaction lock keyed on ``dedup_key`` serialises
    the read-then-write for a given cluster, so two workers processing
    overlapping windows cannot both take the INSERT branch.
    """
    import json

    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtext(:k))"), {"k": payload.dedup_key}
    )
    result = await session.execute(
        _UPSERT_SQL,
        {
            "dedup_key": payload.dedup_key,
            "threat_type": payload.threat_type,
            "count": payload.count,
            "confidence": payload.confidence_score,
            "sectors": payload.sectors,
            "ttl_minutes": payload.ttl_minutes,
            "eta_minutes": payload.eta_minutes,
            "origin_sector": payload.origin_sector,
            "core_ewkb": payload.core_ewkb,
            "risk_ewkb": payload.risk_ewkb,
            "source_channel": payload.source_channel,
            "message_ids": payload.source_message_ids,
            "raw": json.dumps(payload.raw, ensure_ascii=False),
        },
    )
    return str(result.scalar_one())


async def expire_sectors(session: AsyncSession, normalized_sectors: Sequence[str]) -> int:
    """Force-expire live threats whose sector set intersects ``normalized_sectors``.

    Drives the "відбій / all clear" path: an operator announcement retires a
    threat immediately instead of waiting out its TTL.
    """
    if not normalized_sectors:
        return 0
    result = await session.execute(
        text(
            """
            UPDATE active_threats
            SET expires_at = now(), updated_at = now()
            WHERE expires_at > now() AND sectors && CAST(:sectors AS text[])
            """
        ),
        {"sectors": list(normalized_sectors)},
    )
    return int(result.rowcount or 0)


async def purge_expired(session: AsyncSession) -> int:
    """Archive expired clusters into ``threat_history`` and delete them."""
    result = await session.execute(
        text(
            """
            WITH gone AS (
                DELETE FROM active_threats
                WHERE expires_at <= now()
                RETURNING id, dedup_key, threat_type, count, confidence_score,
                          sectors, geom, first_seen_at, raw
            )
            INSERT INTO threat_history (
                id, dedup_key, threat_type, count, confidence_score, sectors,
                geom, first_seen_at, expired_at, raw
            )
            SELECT id, dedup_key, threat_type, count, confidence_score, sectors,
                   geom, first_seen_at, now(), raw
            FROM gone
            RETURNING id
            """
        )
    )
    return len(result.fetchall())


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


#: How many source message ids travel with a threat, out of however many the
#: row has accumulated.
#:
#: The stored array only grows: `upsert_threat` unions it on every re-report,
#: so a cluster reposted for its whole TTL can reach a hundred-odd ids. All of
#: them are worth keeping in PostGIS for audit; none of them are worth putting
#: in a snapshot every API replica decodes, or in a provenance row a phone
#: renders as one line of text. Newest wins — Telegram ids increase with time,
#: so the tail is the part that describes the threat as it is now.
MAX_SOURCE_REFS = 8


@dataclass(slots=True)
class ActiveThreatRecord:
    """Flat, driver-agnostic row used to build the in-process spatial index."""

    id: str
    threat_type: str
    count: int
    confidence_score: float
    sectors: list[str]
    #: The gazetteer's own spelling of the districts under threat
    #: ("Дарницький район"), parallel to `sectors`, which holds the folded
    #: match keys ("дарницький") and is not fit to put on screen. Read back
    #: from `raw.matched_sectors`, and falls back to `sectors` for rows
    #: written before that key existed.
    target_sectors: list[str]
    eta_minutes: int | None
    origin_sector: str | None
    #: When the threat was first reported. Unlike `updated_at`, re-reports do
    #: not move it, so it is what a "time since alert" clock counts from.
    first_seen_at: datetime
    updated_at: datetime
    expires_at: datetime
    geom_wkb: bytes
    core_wkb: bytes | None
    #: Newest first, and at most ``MAX_SOURCE_REFS`` of them — the trim happens
    #: in SQL, so this is not a faithful copy of the column.
    source_message_ids: list[int]


_LOAD_ACTIVE_SQL = text(
    """
    SELECT a.id, a.threat_type, a.count, a.confidence_score, a.sectors,
           a.eta_minutes, a.origin_sector, a.first_seen_at, a.updated_at, a.expires_at,
           COALESCE(
               (
                   SELECT array_agg(s.name ORDER BY ord)
                   FROM jsonb_array_elements_text(a.raw -> 'matched_sectors')
                        WITH ORDINALITY AS s(name, ord)
               ),
               a.sectors
           ) AS target_sectors,
           ST_AsBinary(a.geom)      AS geom_wkb,
           ST_AsBinary(a.core_geom) AS core_wkb,
           COALESCE(
               (
                   SELECT array_agg(newest.mid ORDER BY newest.mid DESC)
                   FROM (
                       SELECT DISTINCT m.mid
                       FROM unnest(a.source_message_ids) AS m(mid)
                       ORDER BY m.mid DESC
                       LIMIT :max_source_refs
                   ) newest
               ),
               '{}'::bigint[]
           ) AS source_message_ids
    FROM active_threats a
    WHERE a.expires_at > now() AND a.confidence_score >= :min_conf
    ORDER BY a.expires_at DESC
    """
)


async def load_active_threats(
    session: AsyncSession, min_confidence: float
) -> list[ActiveThreatRecord]:
    result = await session.execute(
        _LOAD_ACTIVE_SQL,
        {"min_conf": min_confidence, "max_source_refs": MAX_SOURCE_REFS},
    )
    records: list[ActiveThreatRecord] = []
    for row in result.mappings():
        records.append(
            ActiveThreatRecord(
                id=row["id"],
                threat_type=row["threat_type"],
                count=row["count"],
                confidence_score=row["confidence_score"],
                sectors=list(row["sectors"] or []),
                target_sectors=list(row["target_sectors"] or []),
                eta_minutes=row["eta_minutes"],
                origin_sector=row["origin_sector"],
                first_seen_at=row["first_seen_at"],
                updated_at=row["updated_at"],
                expires_at=row["expires_at"],
                geom_wkb=bytes(row["geom_wkb"]),
                core_wkb=bytes(row["core_wkb"]) if row["core_wkb"] is not None else None,
                source_message_ids=[int(m) for m in (row["source_message_ids"] or [])],
            )
        )
    return records


async def find_threats_at_point(
    session: AsyncSession, latitude: float, longitude: float, min_confidence: float
) -> list[dict]:
    """PostGIS fallback for the hot path when no Redis snapshot is available.

    ``ST_Contains`` is index-assisted here: the GiST index on ``geom`` serves
    the bounding-box stage before the exact predicate runs.
    """
    result = await session.execute(
        text(
            """
            SELECT id, threat_type, confidence_score, sectors, eta_minutes, updated_at,
                   ST_Contains(core_geom, pt.g) AS in_core
            FROM active_threats,
                 LATERAL (SELECT ST_SetSRID(ST_MakePoint(:lon, :lat), 4326) AS g) pt
            WHERE expires_at > now()
              AND confidence_score >= :min_conf
              AND ST_Contains(geom, pt.g)
            """
        ),
        {"lat": latitude, "lon": longitude, "min_conf": min_confidence},
    )
    return [dict(row) for row in result.mappings()]


@dataclass(slots=True)
class SectorRecord:
    id: int
    name: str
    name_norm: str
    geom_wkb: bytes


async def load_sectors(session: AsyncSession) -> list[SectorRecord]:
    result = await session.execute(
        text("SELECT id, name, name_norm, ST_AsBinary(geom) AS geom_wkb FROM sectors")
    )
    return [
        SectorRecord(
            id=row["id"],
            name=row["name"],
            name_norm=row["name_norm"],
            geom_wkb=bytes(row["geom_wkb"]),
        )
        for row in result.mappings()
    ]


#: Registers whose rows may be served. The KNN query filters on this, so seeding the
#: table from anywhere else cannot put an unusable shelter in front of a user.
OFFICIAL_SOURCES: frozenset[str] = frozenset({"kmda", "koda"})


@dataclass(slots=True)
class NearestShelter:
    id: str
    name: str
    shelter_type: str | None
    source: str
    address: str | None
    district: str | None
    kind: str | None
    building_type: str | None
    working_time: str | None
    phone: str | None
    accessible: bool | None
    notes: str | None
    #: Distinct devices that have flagged a problem here.
    reports: int
    latitude: float
    longitude: float
    distance_meters: float


#: Index-assisted KNN, then an exact re-sort.
#:
#: ``<->`` on geometry measures planar degrees, and at Kyiv's latitude a degree of
#: longitude is ~0.64 of a degree of latitude — so degree order is not metre order
#: and the "nearest" row can come back second. Only ``<->`` can use the GiST index
#: though, so the fix is to let it choose a generous candidate set and re-rank those
#: few rows by true spheroidal distance.
_NEAREST_SHELTERS_SQL = text(
    """
    WITH pt AS (
        SELECT ST_SetSRID(ST_MakePoint(:lon, :lat), 4326) AS g
    ),
    candidates AS (
        SELECT s.id, s.name, s.shelter_type, s.source, s.address, s.district,
               s.kind, s.building_type, s.working_time, s.phone, s.accessible,
               s.notes, s.geom
        FROM shelters s, pt
        WHERE s.source = ANY(:sources)
        ORDER BY s.geom <-> pt.g
        LIMIT :candidates
    ),
    deduped AS (
        -- The registers list one doorway many times: 85 rows share a single metro
        -- entrance. Without this, `limit=5` can return five copies of one place and
        -- hide the four real alternatives behind it. Snap to ~1 m and keep the entry
        -- worth walking to: open around the clock first, then the city register.
        SELECT DISTINCT ON (ST_SnapToGrid(geom, 0.00001)) *
        FROM candidates
        ORDER BY ST_SnapToGrid(geom, 0.00001),
                 (working_time ILIKE '%цілодобово%') DESC,
                 (source = 'kmda') DESC,
                 id
    )
    SELECT c.id,
           c.name,
           c.shelter_type,
           c.source,
           c.address,
           c.district,
           c.kind,
           c.building_type,
           c.working_time,
           c.phone,
           c.accessible,
           c.notes,
           COALESCE(r.reports, 0) AS reports,
           ST_Y(c.geom) AS latitude,
           ST_X(c.geom) AS longitude,
           ST_DistanceSphere(c.geom, pt.g) AS distance_meters
    FROM deduped c
    CROSS JOIN pt
    -- Reports aggregate by *place*, not by row. The register lists one doorway several
    -- times and the dedupe above picks an arbitrary representative, so a report filed
    -- against one duplicate has to count for the entry everybody else is shown.
    LEFT JOIN (
        SELECT ST_SnapToGrid(s2.geom, 0.00001) AS g,
               count(DISTINCT rep.client_id) AS reports
        FROM shelter_reports rep
        JOIN shelters s2 ON s2.id = rep.shelter_id
        GROUP BY 1
    ) r ON r.g = ST_SnapToGrid(c.geom, 0.00001)
    -- Strictly by distance. An earlier version sank flagged entries, which at limit=5
    -- pushed the *nearest* shelter off the list entirely — three anonymous taps and the
    -- closest doorway disappears. That is the denial-of-service this feature must not
    -- become. The count travels with the row instead, and the app warns; a human
    -- decides whether the entry is actually wrong.
    ORDER BY distance_meters
    LIMIT :limit
    """
)


async def find_nearest_shelters(
    session: AsyncSession, latitude: float, longitude: float, limit: int
) -> list[NearestShelter]:
    result = await session.execute(
        _NEAREST_SHELTERS_SQL,
        {
            "lat": latitude,
            "lon": longitude,
            "limit": limit,
            "candidates": max(limit * 20, 120),
            "sources": list(OFFICIAL_SOURCES),
        },
    )
    return [
        NearestShelter(
            id=row["id"],
            name=row["name"],
            shelter_type=row["shelter_type"],
            source=row["source"],
            address=row["address"],
            district=row["district"],
            kind=row["kind"],
            building_type=row["building_type"],
            working_time=row["working_time"],
            phone=row["phone"],
            accessible=row["accessible"],
            notes=row["notes"],
            reports=int(row["reports"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            distance_meters=float(row["distance_meters"]),
        )
        for row in result.mappings()
    ]


REPORT_REASONS: frozenset[str] = frozenset(
    {"wrong_location", "locked", "does_not_exist", "wrong_hours", "other"}
)

#: How many *different* shelters one device may flag per hour.
#:
#: The unique index stops a device reporting the same shelter twice, but nothing stopped
#: it walking the map and flagging all five thousand. Someone genuinely finding several
#: bad entries in an hour is rare and still fits inside this; someone poisoning the map
#: hits it immediately.
REPORT_RATE_LIMIT_PER_HOUR = 8


class ReportRateLimited(Exception):
    """Raised when one device has flagged too many shelters in the last hour."""


async def record_shelter_report(
    session: AsyncSession, shelter_id: str, reason: str, note: str | None, client_id: str
) -> int:
    """Store a report and return how many distinct devices have flagged this shelter.

    Reports are never allowed to delete or hide a shelter on their own. Anyone can
    post one, and a shelter that quietly disappears because three strangers said so
    is a denial-of-service against the people who need it. The count rides along with
    the shelter so a human can act on it — and so the app can warn honestly.
    """
    exists = (
        await session.execute(
            text("SELECT 1 FROM shelters WHERE id = :id"), {"id": shelter_id}
        )
    ).scalar_one_or_none()
    if exists is None:
        raise LookupError(shelter_id)

    # Count distinct shelters, not rows: re-reporting one entry with a second reason is
    # a person adding detail, not abuse, and should not spend the budget.
    recent = int(
        (
            await session.execute(
                text(
                    """
                    SELECT count(DISTINCT shelter_id) FROM shelter_reports
                    WHERE client_id = :client_id
                      AND created_at > now() - interval '1 hour'
                      AND shelter_id <> :shelter_id
                    """
                ),
                {"client_id": client_id, "shelter_id": shelter_id},
            )
        ).scalar_one()
    )
    if recent >= REPORT_RATE_LIMIT_PER_HOUR:
        raise ReportRateLimited(client_id)

    await session.execute(
        text(
            """
            INSERT INTO shelter_reports (id, shelter_id, reason, note, client_id)
            VALUES (gen_random_uuid()::text, :shelter_id, :reason, :note, :client_id)
            ON CONFLICT (shelter_id, client_id, reason)
            DO UPDATE SET note = EXCLUDED.note, created_at = now()
            """
        ),
        {"shelter_id": shelter_id, "reason": reason, "note": note, "client_id": client_id},
    )
    # Count for the place, so the number we hand back is the one other people will
    # see on the entry the API actually returns for this doorway.
    return int(
        (
            await session.execute(
                text(
                    """
                    SELECT count(DISTINCT rep.client_id)
                    FROM shelter_reports rep
                    JOIN shelters s2 ON s2.id = rep.shelter_id
                    WHERE ST_SnapToGrid(s2.geom, 0.00001) = (
                        SELECT ST_SnapToGrid(geom, 0.00001) FROM shelters WHERE id = :id
                    )
                    """
                ),
                {"id": shelter_id},
            )
        ).scalar_one()
    )
