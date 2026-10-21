"""End-to-end pipeline check that needs no API keys.

Exercises everything between "the model returned a threat" and "the API answers
a coordinate": sector resolution, PostGIS geometry construction, the
deduplicating upsert, the Redis snapshot, all-clear expiry and the janitor.
The LLM and Telegram are the only parts it does not touch.

    docker compose exec worker python scripts/smoke_test.py

Safe to re-run: it cleans up the rows it creates. Run it after changing
anything in geo.py, database.py or cache.py.
"""

from __future__ import annotations

import asyncio
import math
import sys
import time
import uuid

from sqlalchemy import text

from app import cache, database, geo
from app.config import get_settings
from app.database import MAX_SOURCE_REFS, ThreatUpsert
from app.schemas import ExtractedThreat, ThreatCheckResponse
from app.spatial import ThreatIndex, format_eta_window

PASS, FAIL = "  ok  ", " FAIL "
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{PASS if condition else FAIL}] {label}{f' — {detail}' if detail else ''}")
    if not condition:
        failures.append(label)


async def main() -> int:
    settings = get_settings()
    # Everything this run writes is tagged with a unique id. Production ingestion
    # is live while this runs: without it, an identical real shahed report would
    # share our dedup_key and the upsert would mutate a real threat, and our
    # all-clear would expire every real threat in the same district.
    run_id = uuid.uuid4().hex[:8]
    marker_sector = f"smoketest-{run_id}"
    engine = database.create_engine(settings)
    sessionmaker = database.create_sessionmaker(engine)
    redis = cache.create_redis(settings.redis_url)

    threat = ExtractedThreat(
        threat_type="shahed",
        count=5,
        target_sectors=["Оболонський район"],
        confidence_score=0.85,
        ttl_minutes=20,
        eta_minutes=10,
        origin_sector="Київська область",
        source_refs=[0],
    )

    print("\n=== 1. sector resolution ===")
    async with database.session_scope(sessionmaker) as session:
        exact = await geo.resolve_sectors(
            session, ["Оболонський район"], settings.sector_match_threshold
        )
        check("exact name match", exact.matched_names == ["Оболонський район"], str(exact.matched_names))

        alias = await geo.resolve_sectors(session, ["Оболонь"], settings.sector_match_threshold)
        check("alias match", alias.matched_names == ["Оболонський район"], str(alias.matched_names))

        fuzzy = await geo.resolve_sectors(
            session, ["оболонський р-н"], settings.sector_match_threshold
        )
        check("trigram fuzzy match", bool(fuzzy.sector_ids), str(fuzzy.matched_names))

        coords = await geo.resolve_sectors(session, ["50.4501,30.5234"], settings.sector_match_threshold)
        check("raw coordinates parsed", coords.points == [(50.4501, 30.5234)], str(coords.points))

        junk = await geo.resolve_sectors(session, ["Атлантида"], settings.sector_match_threshold)
        check("unknown name rejected", not junk.is_localisable, f"unmatched={junk.unmatched}")

    print("\n=== 2. PostGIS geometry construction ===")
    async with database.session_scope(sessionmaker) as session:
        resolved = await geo.resolve_sectors(
            session, threat.target_sectors, settings.sector_match_threshold
        )
        origin = await geo.resolve_origin(
            session, threat.origin_sector, settings.sector_match_threshold
        )
        check("corridor origin resolved", origin is not None, f"origin={origin}")
        if origin is None:
            return 1
        # The name is what `/threats/check` shows as "approaching from", so it
        # has to be the gazetteer's spelling and not the model's.
        check(
            "origin resolves to a gazetteer name",
            origin.name == "Київська область",
            origin.name,
        )

        built = await geo.build_risk_geometry(
            session,
            resolved,
            buffer_ratio=settings.buffer_ratio_for("shahed"),
            buffer_floor_meters=settings.buffer_floor_meters,
            buffer_ceiling_meters=settings.buffer_ceiling_meters,
            point_radius_meters=settings.point_radius_meters,
            origin_sector_id=origin.sector_id,
        )
        check("geometry built", built is not None)
        if built is None:
            return 1
        core_ewkb, risk_ewkb = built.core_ewkb, built.risk_ewkb

        # The risk area must strictly contain the core, and the buffer must be
        # real metres — a degrees-vs-metres mistake shows up here as a ~100x
        # area error.
        row = (
            await session.execute(
                text(
                    """
                    SELECT ST_Area(ST_GeomFromEWKB(:core)::geography) / 1e6  AS core_km2,
                           ST_Area(ST_GeomFromEWKB(:risk)::geography) / 1e6  AS risk_km2,
                           ST_Contains(ST_GeomFromEWKB(:risk), ST_GeomFromEWKB(:core)) AS contains
                    """
                ),
                {"core": core_ewkb, "risk": risk_ewkb},
            )
        ).mappings().one()
        check("risk area contains core", bool(row["contains"]))
        # The buffer scales with the target, so a fixed area band would have to
        # be re-tuned for whatever district the gazetteer resolves to. Assert
        # the rule instead: the margin is `ratio * sqrt(core area)`, clamped.
        expected_buffer = min(
            settings.buffer_ceiling_meters,
            max(
                settings.buffer_floor_meters,
                settings.buffer_ratio_for("shahed") * math.sqrt(row["core_km2"] * 1e6),
            ),
        )
        check(
            "buffer scales with target size",
            abs(built.buffer_meters - expected_buffer) < 1.0,
            f"applied={built.buffer_meters:.0f} m, expected={expected_buffer:.0f} m "
            f"(core={row['core_km2']:.0f} km²)",
        )
        # A degrees-vs-metres slip shows up as a ~100x area error in either
        # direction, which no plausible buffer can produce.
        check(
            "buffer is metres, not degrees",
            1.5 < row["risk_km2"] / row["core_km2"] < 25,
            f"core={row['core_km2']:.0f} km², risk={row['risk_km2']:.0f} km², "
            f"ratio={row['risk_km2'] / row['core_km2']:.1f}x",
        )

    print("\n=== 3. deduplicating upsert ===")
    payload = ThreatUpsert(
        dedup_key=f"{threat.dedup_key()}-{run_id}",
        threat_type="shahed",
        count=5,
        confidence_score=0.85,
        # Real resolved districts drive the geometry; the marker is what the
        # all-clear below targets, so it can never touch a production row.
        sectors=[*resolved.normalized_names, marker_sector],
        ttl_minutes=20,
        eta_minutes=10,
        origin_sector=origin.name,
        core_ewkb=core_ewkb,
        risk_ewkb=risk_ewkb,
        source_channel="smoketest",
        source_message_ids=[101],
        raw={"smoke": True},
    )
    async with database.session_scope(sessionmaker) as session:
        first_id = await database.upsert_threat(session, payload)
    check("insert returns id", bool(first_id), first_id)

    # Same cluster reported again with a higher count and a new message id.
    payload.count = 9
    payload.confidence_score = 0.70  # lower — must not overwrite the higher one
    payload.source_message_ids = [202]
    # A window that mentions no approach direction is silence, not a retraction.
    payload.origin_sector = None
    async with database.session_scope(sessionmaker) as session:
        second_id = await database.upsert_threat(session, payload)
    check("re-report updates, does not insert", first_id == second_id)

    async with database.session_scope(sessionmaker) as session:
        row = (
            await session.execute(
                text(
                    "SELECT count, confidence_score, source_message_ids, origin_sector "
                    "FROM active_threats WHERE id = :id"
                ),
                {"id": first_id},
            )
        ).mappings().one()
    check("count takes the maximum", row["count"] == 9, f"count={row['count']}")
    check(
        "confidence takes the maximum",
        abs(row["confidence_score"] - 0.85) < 1e-6,
        f"confidence={row['confidence_score']}",
    )
    check(
        "source ids accumulate",
        sorted(row["source_message_ids"]) == [101, 202],
        str(row["source_message_ids"]),
    )
    check(
        "origin survives a re-report that omits it",
        row["origin_sector"] == origin.name,
        f"origin_sector={row['origin_sector']!r}",
    )

    # Enough ids to overrun the wire cap, so the trim in `load_active_threats`
    # is exercised rather than assumed. A cluster reposted for its whole TTL
    # accumulates this many for real.
    burst = list(range(900, 900 + MAX_SOURCE_REFS + 5))
    payload.source_message_ids = burst
    async with database.session_scope(sessionmaker) as session:
        await database.upsert_threat(session, payload)
        stored_ids = (
            await session.execute(
                text("SELECT source_message_ids FROM active_threats WHERE id = :id"),
                {"id": first_id},
            )
        ).scalar_one()
    check(
        "postgis keeps every source id",
        sorted(stored_ids) == sorted([101, 202, *burst]),
        f"{len(stored_ids)} id(s) stored",
    )

    print("\n=== 4. spatial query + snapshot ===")
    async with database.session_scope(sessionmaker) as session:
        inside = await database.find_threats_at_point(session, 50.54, 30.51, settings.min_confidence)
        outside = await database.find_threats_at_point(session, 46.48, 30.72, settings.min_confidence)
    # Identity, not count: other threats may legitimately be live when this
    # runs against a working system.
    mine = next((hit for hit in inside if hit["id"] == first_id), None)
    check("ST_Contains finds the threat", mine is not None, f"{len(inside)} hit(s) total")
    check("point in core flagged high", bool(mine and mine["in_core"]))
    check(
        "distant point does not match this threat",
        all(hit["id"] != first_id for hit in outside),
        f"{len(outside)} unrelated hit(s) there",
    )

    async with database.session_scope(sessionmaker) as session:
        records = await database.load_active_threats(session, settings.min_confidence)
    check("snapshot rows loaded", len(records) >= 1, f"{len(records)} row(s)")

    record = next((r for r in records if r.id == first_id), None)
    check("this run's row is in the load", record is not None)
    if record is None:
        return 1
    check(
        "origin_sector loads",
        record.origin_sector == origin.name,
        f"origin_sector={record.origin_sector!r}",
    )
    check(
        "source ids are trimmed to the wire cap, newest first",
        record.source_message_ids == sorted(burst, reverse=True)[:MAX_SOURCE_REFS],
        f"{record.source_message_ids} (cap={MAX_SOURCE_REFS})",
    )

    version = await cache.publish_snapshot(redis, records)
    check("snapshot published to redis", version > 0, f"version={version}")
    round_tripped = await cache.get_snapshot(redis)
    check("snapshot decodes", round_tripped is not None and len(round_tripped) == len(records))

    # The iOS contract, end to end. `eta_minutes`, `origin_sector` and
    # `source_refs` each have to survive PostGIS -> Redis -> ThreatIndex ->
    # response; a break anywhere in that chain shows up as a permanently
    # invisible row on the dashboard rather than as an error, so assert on the
    # serialised payload and not on the intermediates.
    #
    # The index is built from this run's threat alone: `lookup` returns the
    # single most severe match, and real ingestion can legitimately have a
    # missile over Obolon while this runs.
    snapshot_rows = [t for t in (round_tripped or []) if t.id == first_id]
    check(
        "this run's threat round-trips",
        len(snapshot_rows) == 1,
        f"{len(snapshot_rows)} match(es)",
    )
    match = (
        ThreatIndex(snapshot_rows).lookup(50.54, 30.51, settings.high_confidence)
        if snapshot_rows
        else None
    )
    check("index lookup finds it", match is not None)
    if match is None:
        return 1

    # Constructed the way `main.check_threats` constructs it.
    payload_json = ThreatCheckResponse(
        in_risk_zone=True,
        risk_level=match.risk_level,
        threat_type=match.threat_type,
        eta_window=format_eta_window(match.eta_minutes, match.updated_at),
        eta_minutes=match.eta_minutes,
        sector_name=None,
        origin_sector=match.origin_sector,
        source_refs=match.source_message_ids,
    ).model_dump()
    check(
        "response carries eta_minutes",
        payload_json["eta_minutes"] == 10,
        f"eta_minutes={payload_json['eta_minutes']!r}",
    )
    check(
        "eta_window still derives from the same number",
        payload_json["eta_window"] is not None,
        f"eta_window={payload_json['eta_window']!r}",
    )
    check(
        "response carries origin_sector",
        payload_json["origin_sector"] == origin.name,
        f"origin_sector={payload_json['origin_sector']!r}",
    )
    check(
        "response carries source_refs",
        payload_json["source_refs"] == sorted(burst, reverse=True)[:MAX_SOURCE_REFS],
        str(payload_json["source_refs"]),
    )

    print("\n=== 5. all-clear + janitor ===")
    async with database.session_scope(sessionmaker) as session:
        cleared = await database.expire_sectors(session, [marker_sector])
    # Exactly one: the marker sector belongs to this run alone.
    check("all-clear expires the threat", cleared == 1, f"{cleared} row(s)")

    async with database.session_scope(sessionmaker) as session:
        still_live = await database.find_threats_at_point(session, 50.54, 30.51, settings.min_confidence)
    # Live threats from real ingestion may legitimately cover this point; the
    # assertion is that *ours* stopped matching, not that the sky is empty.
    check(
        "expired threat no longer matches",
        all(hit["id"] != first_id for hit in still_live),
        f"{len(still_live)} unrelated threat(s) still live there",
    )

    async with database.session_scope(sessionmaker) as session:
        purged = await database.purge_expired(session)
        remaining = (
            await session.execute(
                text("SELECT count(*) FROM active_threats WHERE id = :id"), {"id": first_id}
            )
        ).scalar_one()
    check("janitor archives expired rows", purged >= 1, f"{purged} row(s)")
    check("janitor removed this run's row", remaining == 0)

    async with database.session_scope(sessionmaker) as session:
        archived = (
            await session.execute(
                text("SELECT count(*) FROM threat_history WHERE id = :id"), {"id": first_id}
            )
        ).scalar_one()
        # Leave the database as we found it.
        await session.execute(
            text("DELETE FROM threat_history WHERE id = :id"), {"id": first_id}
        )
    check("row landed in threat_history", archived == 1)

    async with database.session_scope(sessionmaker) as session:
        records = await database.load_active_threats(session, settings.min_confidence)
    await cache.publish_snapshot(redis, records)

    print("\n=== 6. shelters (spatial KNN) ===")
    # Maidan Nezalezhnosti — the coordinate the iOS client uses for its own checks.
    lat, lon = 50.4501, 30.5234
    async with database.session_scope(sessionmaker) as session:
        total_shelters = (
            await session.execute(text("SELECT count(*) FROM shelters"))
        ).scalar_one()
        check("shelters table populated", total_shelters > 0, f"{total_shelters} row(s)")

        # Warm the plan and the buffer cache, then take the median of several
        # runs: this times a round trip between containers, not just the query,
        # and a single sample of that is mostly scheduler noise.
        await database.find_nearest_shelters(session, lat, lon, 5)
        samples: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            nearest = await database.find_nearest_shelters(session, lat, lon, 5)
            samples.append((time.perf_counter() - started) * 1000)
        elapsed_ms = sorted(samples)[len(samples) // 2]

    check("KNN returns the requested count", len(nearest) == 5, f"{len(nearest)} row(s)")
    check(
        "results are sorted by true distance",
        all(a.distance_meters <= b.distance_meters for a, b in zip(nearest, nearest[1:])),
    )
    check(
        "nearest shelter is plausibly close",
        bool(nearest) and nearest[0].distance_meters < 5_000,
        f"{nearest[0].distance_meters:.0f} m" if nearest else "no rows",
    )
    check("every row carries a name", all(s.name for s in nearest))
    # `Execution Time` from the server, with per-node timing off so the measurement
    # does not dominate the thing measured. The client round trip above crosses two
    # containers on a macOS VM and says more about Docker than about the index.
    async with database.session_scope(sessionmaker) as session:
        plan_rows = (
            await session.execute(
                text(
                    "EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON) "
                    "SELECT id FROM shelters "
                    "ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326) LIMIT 5"
                ),
                {"lat": lat, "lon": lon},
            )
        ).scalar_one()
    server_ms = float(plan_rows[0]["Execution Time"])
    check(
        f"KNN executes under 5 ms server-side ({server_ms:.2f} ms)",
        server_ms < 5.0,
        f"client round trip median {elapsed_ms:.2f} ms across containers",
    )

    # The index has to be doing the work — a sequential scan would still pass the
    # assertions above on a small table and fall over on a national extract.
    async with database.session_scope(sessionmaker) as session:
        plan = (
            await session.execute(
                text(
                    "EXPLAIN (FORMAT TEXT) "
                    "SELECT id FROM shelters "
                    "ORDER BY geom <-> ST_SetSRID(ST_MakePoint(:lon, :lat), 4326) LIMIT 5"
                ),
                {"lat": lat, "lon": lon},
            )
        ).scalars().all()
    check("KNN uses the GiST index", any("Index Scan" in line for line in plan), plan[0].strip())


    await redis.aclose()
    await engine.dispose()

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("ALL PIPELINE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
