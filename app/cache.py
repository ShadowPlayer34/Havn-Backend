"""Redis: the fan-out channel between the worker and every API replica.

The worker publishes a full snapshot of the active threat set after each
batching window; API processes poll a monotonically increasing version key and
rebuild their in-process spatial index only when it changes. Steady state is one
4-byte ``GET`` every ``SNAPSHOT_POLL_SECONDS`` per replica — the request path
itself does no I/O at all.

The snapshot is a *derived* artifact. Losing Redis entirely degrades the API to
a direct PostGIS query (see ``database.find_threats_at_point``), it does not
break it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import orjson
from redis.asyncio import Redis

from app.database import ActiveThreatRecord

logger = logging.getLogger(__name__)

SNAPSHOT_KEY = "alertradar:threats:snapshot"
SNAPSHOT_VERSION_KEY = "alertradar:threats:snapshot:version"

#: Safety net — a stale snapshot is worse than no snapshot, so let Redis drop it
#: if the worker dies. Comfortably longer than any legitimate quiet period.
SNAPSHOT_TTL_SECONDS = 900


@dataclass(slots=True)
class SnapshotThreat:
    id: str
    threat_type: str
    count: int
    confidence_score: float
    sectors: list[str]
    #: Display spelling of `sectors`; see `database.ActiveThreatRecord`.
    target_sectors: list[str]
    eta_minutes: int | None
    origin_sector: str | None
    first_seen_at: datetime
    updated_at: datetime
    expires_at: datetime
    geom_wkb: bytes
    core_wkb: bytes | None
    #: Already trimmed to ``database.MAX_SOURCE_REFS``, newest first.
    source_message_ids: list[int]


def create_redis(url: str) -> Redis:
    return Redis.from_url(url, decode_responses=False, health_check_interval=30)


def _encode(records: list[ActiveThreatRecord]) -> bytes:
    return orjson.dumps(
        [
            {
                "id": r.id,
                "threat_type": r.threat_type,
                "count": r.count,
                "confidence_score": r.confidence_score,
                "sectors": r.sectors,
                "target_sectors": r.target_sectors,
                "eta_minutes": r.eta_minutes,
                "origin_sector": r.origin_sector,
                "source_message_ids": r.source_message_ids,
                "first_seen_at": r.first_seen_at.isoformat(),
                "updated_at": r.updated_at.isoformat(),
                "expires_at": r.expires_at.isoformat(),
                "geom": r.geom_wkb.hex(),
                "core": r.core_wkb.hex() if r.core_wkb else None,
            }
            for r in records
        ]
    )


def _decode(payload: bytes) -> list[SnapshotThreat]:
    items = orjson.loads(payload)
    return [
        SnapshotThreat(
            id=item["id"],
            threat_type=item["threat_type"],
            count=item["count"],
            confidence_score=item["confidence_score"],
            sectors=item["sectors"],
            eta_minutes=item["eta_minutes"],
            # `.get` rather than `[]` for the fields added after this format
            # shipped. During a rolling upgrade the API comes up against a
            # snapshot the old worker wrote; a KeyError here would discard the
            # whole payload and blank the threat index until the worker's next
            # window, which is the one moment that must not happen.
            origin_sector=item.get("origin_sector"),
            target_sectors=item.get("target_sectors") or item["sectors"],
            source_message_ids=item.get("source_message_ids") or [],
            # A snapshot from before the field knows only the last report, which
            # is the closest it can say to when the threat began.
            first_seen_at=datetime.fromisoformat(
                item.get("first_seen_at") or item["updated_at"]
            ),
            updated_at=datetime.fromisoformat(item["updated_at"]),
            expires_at=datetime.fromisoformat(item["expires_at"]),
            geom_wkb=bytes.fromhex(item["geom"]),
            core_wkb=bytes.fromhex(item["core"]) if item["core"] else None,
        )
        for item in items
    ]


async def publish_snapshot(redis: Redis, records: list[ActiveThreatRecord]) -> int:
    """Write the snapshot and bump its version. Returns the new version."""
    pipe = redis.pipeline(transaction=True)
    pipe.set(SNAPSHOT_KEY, _encode(records), ex=SNAPSHOT_TTL_SECONDS)
    pipe.incr(SNAPSHOT_VERSION_KEY)
    pipe.expire(SNAPSHOT_VERSION_KEY, SNAPSHOT_TTL_SECONDS)
    results = await pipe.execute()
    version = int(results[1])
    logger.info("snapshot_published threats=%d version=%d", len(records), version)
    return version


async def get_snapshot_version(redis: Redis) -> int | None:
    raw = await redis.get(SNAPSHOT_VERSION_KEY)
    return int(raw) if raw is not None else None


async def get_snapshot(redis: Redis) -> list[SnapshotThreat] | None:
    raw = await redis.get(SNAPSHOT_KEY)
    if raw is None:
        return None
    try:
        return _decode(raw)
    except (orjson.JSONDecodeError, KeyError, ValueError):
        logger.exception("snapshot_decode_failed — treating as absent")
        return None
