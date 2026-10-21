"""Inject demo threats over Kyiv so the client can be shown in each alert state.

    python scripts/seed_demo_threats.py both      # red + yellow at once
    python scripts/seed_demo_threats.py red       # high  -> red
    python scripts/seed_demo_threats.py yellow    # elevated -> yellow
    python scripts/seed_demo_threats.py clear     # remove demo rows

Everything goes through the worker's own persistence path, so the geometry is
what a real extraction would have produced — not a hand-built polygon.

Rows are tagged ``source_channel='__demo__'``. ``clear`` hard-deletes exactly
those rows, which also keeps them out of ``threat_history``: letting them
expire would archive fake clusters into the audit table.

The two levels come from different mechanisms on purpose:

* RED    — inside the core polygon AND confidence >= ``high_confidence`` (0.65).
* YELLOW — either outside the core but inside the buffer/corridor, or inside
  the core with confidence below that threshold. The second case is the one a
  buffer-only demo never shows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request

from sqlalchemy import text

from app import cache, database, geo
from app.config import get_settings
from app.database import ThreatUpsert
from app.schemas import ExtractedThreat

DEMO_CHANNEL = "__demo__"
API_BASE = "http://api:8000/api/v1"

RED = dict(
    threat_type="missile",
    count=3,
    target_sectors=["Шевченківський район"],
    confidence_score=0.92,
    eta_minutes=7,
    origin_sector="Броварський район",
)
YELLOW = dict(
    threat_type="shahed",
    count=5,
    target_sectors=["Дарницький район"],
    confidence_score=0.50,
    eta_minutes=18,
    origin_sector="Броварський район",
)
SCENARIOS = {"red": [RED], "yellow": [YELLOW], "both": [RED, YELLOW], "clear": []}

# Guaranteed-inside points (ST_PointOnSurface of each district polygon).
PROBES = [
    ("Шевченківський", 50.4629, 30.4518),
    ("Дарницький", 50.4026, 30.7099),
    ("Печерський", 50.4287, 30.5490),
    ("Оболонський", 50.5306, 30.4384),
    ("Львів (контроль)", 49.8397, 24.0297),
]
LABEL = {"high": "RED", "elevated": "YELLOW", "none": "none"}


async def seed(scenario: str, ttl_minutes: int) -> None:
    settings = get_settings()
    engine = database.create_engine(settings)
    sessionmaker = database.create_sessionmaker(engine)
    redis = cache.create_redis(settings.redis_url)

    async with database.session_scope(sessionmaker) as session:
        removed = (
            await session.execute(
                text("DELETE FROM active_threats WHERE source_channel = :c"),
                {"c": DEMO_CHANNEL},
            )
        ).rowcount
        if removed:
            print(f"cleared {removed} previous demo threat(s)")

        for spec in SCENARIOS[scenario]:
            threat = ExtractedThreat(ttl_minutes=ttl_minutes, **spec)
            resolved = await geo.resolve_sectors(
                session, threat.target_sectors, settings.sector_match_threshold
            )
            if not resolved.is_localisable:
                print(f"  !! unlocalised: {threat.target_sectors}")
                continue
            origin = await geo.resolve_origin(
                session, threat.origin_sector, settings.sector_match_threshold
            )
            geometry = await geo.build_risk_geometry(
                session,
                resolved,
                buffer_ratio=settings.buffer_ratio_for(threat.threat_type),
                buffer_floor_meters=settings.buffer_floor_meters,
                buffer_ceiling_meters=settings.buffer_ceiling_meters,
                point_radius_meters=settings.point_radius_meters,
                origin_sector_id=origin.sector_id if origin else None,
            )
            if geometry is None:
                print(f"  !! geometry failed: {threat.target_sectors}")
                continue

            await database.upsert_threat(
                session,
                ThreatUpsert(
                    dedup_key=threat.dedup_key(),
                    threat_type=str(threat.threat_type),
                    count=threat.count,
                    confidence_score=threat.confidence_score,
                    sectors=resolved.normalized_names,
                    ttl_minutes=ttl_minutes,
                    eta_minutes=threat.eta_minutes,
                    origin_sector=origin.name if origin else None,
                    core_ewkb=geometry.core_ewkb,
                    risk_ewkb=geometry.risk_ewkb,
                    source_channel=DEMO_CHANNEL,
                    source_message_ids=[999_000_001, 999_000_002],
                    # `matched_sectors` is what `target_sectors` is read back
                    # from — without it the API falls back to the folded keys
                    # and the demo misrepresents what production shows.
                    raw={
                        "demo": True,
                        "extracted": threat.model_dump(mode="json"),
                        "matched_sectors": resolved.matched_names,
                    },
                ),
            )
            expect = "RED" if threat.confidence_score >= settings.high_confidence else "YELLOW"
            print(
                f"  + {expect:<6} {threat.threat_type:<8} "
                f"{resolved.matched_names[0]:<22} conf={threat.confidence_score} "
                f"eta={threat.eta_minutes}m ttl={ttl_minutes}m"
            )

        live = (
            await session.execute(
                text(
                    "SELECT threat_type, sectors, confidence_score, source_channel "
                    "FROM active_threats "
                    "WHERE source_channel IS DISTINCT FROM :c AND expires_at > now()"
                ),
                {"c": DEMO_CHANNEL},
            )
        ).all()

    async with database.session_scope(sessionmaker) as session:
        records = await database.load_active_threats(session, settings.min_confidence)
    version = await cache.publish_snapshot(redis, records)
    print(f"snapshot published: threats={len(records)} version={version}")

    if live:
        print("\nNOTE — real threats from the live feed are also active:")
        for t, sectors, conf, chan in live:
            print(f"  · {t} {list(sectors)} conf={conf} from {chan}")
        print("  Stop the worker to demo against demo data only.")

    await redis.aclose()
    await engine.dispose()


def verify() -> None:
    print(f"\n{'point':<22}{'level':<9}{'type':<9}{'ETA':<6}sector")
    print("-" * 66)
    for label, lat, lon in PROBES:
        url = f"{API_BASE}/threats/check?latitude={lat}&longitude={lon}"
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.load(r)
        eta = d.get("eta_minutes")
        print(
            f"{label:<22}{LABEL.get(d['risk_level'], d['risk_level']):<9}"
            f"{str(d.get('threat_type') or '-'):<9}"
            f"{str(eta if eta is not None else '-'):<6}"
            f"{d.get('sector_name') or '-'}"
        )

    with urllib.request.urlopen(f"{API_BASE}/threats/zones", timeout=10) as r:
        zones = json.load(r)
    red = sum(
        1
        for f in zones["features"]
        if f["properties"]["kind"] == "core" and f["properties"].get("direct_hit")
    )
    yellow = sum(1 for f in zones["features"] if f["properties"]["kind"] == "risk")
    print(f"\nmap overlay: {red} red core polygon(s), {yellow} yellow risk polygon(s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("scenario", choices=sorted(SCENARIOS), nargs="?", default="both")
    p.add_argument("--ttl", type=int, default=60, help="minutes before auto-expiry")
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    asyncio.run(seed(args.scenario, args.ttl))
    if not args.no_verify:
        # The API polls the snapshot version every `snapshot_poll_seconds` (2s).
        import time

        time.sleep(4)
        verify()


if __name__ == "__main__":
    main()
