"""Payload budget check for the map overlay and the hot path.

Answers one question: **can the client actually receive this in time?** The iOS
app gives `/threats/check` 4 s and `/threats/zones` 8 s (`ThreatAPI` in
`ThreatNetworkService.swift`), and the night those timeouts matter is the night
half the country is under alert and the network is congested. So the sizes here
are measured against escalating threat loads, not against a quiet Tuesday.

    docker compose exec worker python scripts/bench_payload.py

Read-only: geometry is built from the live gazetteer, assembled into an
in-memory `ThreatIndex`, and serialised through the same code the API uses. It
writes nothing and does not touch Redis, so it is safe to run against a live
deployment.
"""

from __future__ import annotations

import asyncio
import gzip
import sys
import time
from datetime import UTC, datetime, timedelta

import orjson
import shapely
from shapely import wkb
from sqlalchemy import text

from app import cache, database, geo
from app.config import get_settings
from app.schemas import ThreatCheckResponse
from app.spatial import ThreatIndex, threat_geometry_geojson, threats_geojson

PASS, FAIL = "  ok  ", " FAIL "
failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"[{PASS if condition else FAIL}] {label}{f' — {detail}' if detail else ''}")
    if not condition:
        failures.append(label)


# Effective throughput and round-trip time, not headline speeds. A Ukrainian
# phone on a congested mast during an alert is the case worth sizing for; LTE
# is here as the "everything is fine" reference, not as the design target.
#: (label, kilobits/s, round-trip ms)
LINKS: list[tuple[str, float, float]] = [
    ("2G / EDGE", 50, 700),
    ("slow 3G", 400, 400),
    ("3G", 1_600, 200),
    ("LTE", 10_000, 60),
]

#: Connection setup before the first byte: TCP + TLS is ~3 RTT on a cold
#: connection. The app reuses a `URLSession`, so a warm poll pays ~1 RTT — the
#: pessimistic figure is the one that has to fit the timeout.
_COLD_RTTS = 3


def count_positions(coordinates: object) -> int:
    """Vertices in a GeoJSON coordinate array, at any nesting depth."""
    if not isinstance(coordinates, list):
        return 0
    if coordinates and isinstance(coordinates[0], (int, float)):
        return 1
    return sum(count_positions(item) for item in coordinates)


def transfer_seconds(payload_bytes: int, kbps: float, rtt_ms: float) -> float:
    return _COLD_RTTS * rtt_ms / 1000.0 + (payload_bytes * 8) / (kbps * 1000.0)


def report_timings(label: str, payload_bytes: int, budget_seconds: float) -> None:
    """Print time-to-last-byte per link profile and flag anything over budget."""
    worst = ""
    for link, kbps, rtt in LINKS:
        seconds = transfer_seconds(payload_bytes, kbps, rtt)
        marker = "  <-- over budget" if seconds > budget_seconds else ""
        print(f"        {link:<12}{seconds:>7.2f} s{marker}")
        if seconds > budget_seconds and not worst:
            worst = link
    check(
        f"{label} fits the {budget_seconds:g} s client timeout on every link",
        not worst,
        f"first failure: {worst}" if worst else f"{payload_bytes / 1024:.1f} KiB",
    )


# Escalating load. `sectors` is resolved against the live gazetteer, so this
# works for whatever regions are actually loaded.
SCENARIOS: list[tuple[str, int, str]] = [
    ("quiet night", 1, "district"),
    ("typical alert", 3, "district"),
    ("heavy night", 8, "district"),
    ("national barrage", 12, "oblast"),
]


async def _sector_names(session, kind: str, limit: int) -> list[str]:
    """Real sector names, largest-first for oblasts and smallest-first for
    districts — the two ends of the size range the overlay has to carry."""
    order = "DESC" if kind == "oblast" else "ASC"
    rows = (
        await session.execute(
            text(
                f"""
                SELECT name FROM sectors
                WHERE geom IS NOT NULL
                ORDER BY ST_Area(geom) {order}
                LIMIT :limit
                """
            ),
            {"limit": limit},
        )
    ).scalars()
    return list(rows)


async def main() -> int:
    settings = get_settings()
    engine = database.create_engine(settings)
    sessionmaker = database.create_sessionmaker(engine)
    now = datetime.now(UTC)

    print(f"budget: {settings.zones_max_vertices} vertices/shape, "
          f"tolerance {settings.zones_simplify_degrees}°, "
          f"precision {settings.zones_precision} dp")

    async with database.session_scope(sessionmaker) as session:
        for label, cluster_count, kind in SCENARIOS:
            names = await _sector_names(session, kind, cluster_count)
            if len(names) < cluster_count:
                print(f"\n=== {label} — skipped, gazetteer has too few sectors ===")
                continue

            print(f"\n=== {label}: {cluster_count} clusters ({kind}-scale) ===")
            snapshot: list[cache.SnapshotThreat] = []
            raw_vertices = 0
            for position, name in enumerate(names):
                threat_type = "missile" if kind == "oblast" else "shahed"
                resolved = await geo.resolve_sectors(
                    session, [name], settings.sector_match_threshold
                )
                if not resolved.is_localisable:
                    continue
                built = await geo.build_risk_geometry(
                    session,
                    resolved,
                    buffer_ratio=settings.buffer_ratio_for(threat_type),
                    buffer_floor_meters=settings.buffer_floor_meters,
                    buffer_ceiling_meters=settings.buffer_ceiling_meters,
                    point_radius_meters=settings.point_radius_meters,
                    origin_sector_id=None,
                )
                if built is None:
                    continue
                raw_vertices += shapely.get_num_coordinates(wkb.loads(built.core_ewkb))
                raw_vertices += shapely.get_num_coordinates(wkb.loads(built.risk_ewkb))
                snapshot.append(
                    cache.SnapshotThreat(
                        id=f"bench-{position}",
                        threat_type=threat_type,
                        count=6,
                        confidence_score=0.85,
                        sectors=[name],
                        target_sectors=[name],
                        eta_minutes=12,
                        origin_sector=None,
                        first_seen_at=now,
                        updated_at=now,
                        expires_at=now + timedelta(minutes=20),
                        geom_wkb=built.risk_ewkb,
                        core_wkb=built.core_ewkb,
                        source_message_ids=[],
                    )
                )

            if not snapshot:
                print("    no geometry built — skipping")
                continue

            index = ThreatIndex(snapshot, version=1)

            started = time.perf_counter()
            payload = threats_geojson(
                index,
                settings.zones_simplify_degrees,
                settings.zones_precision,
                settings.zones_max_vertices,
                now=now,
            )
            serialise_ms = (time.perf_counter() - started) * 1000

            # GZipMiddleware's own level, so this is the byte count on the wire
            # for any client sending `Accept-Encoding: gzip` — which URLSession
            # always does.
            gzipped = gzip.compress(payload, compresslevel=9)

            # Vertex budget, measured on the bytes that actually shipped
            # rather than on a re-simplified copy — the payload is the only
            # thing the client's decoder has to chew through.
            counts = [
                (f["properties"]["kind"], count_positions(f["geometry"]["coordinates"]))
                for f in orjson.loads(payload)["features"]
            ]
            over = [c for c in counts if c[1] > settings.zones_max_vertices]

            print(f"    raw vertices        {raw_vertices:>8,}")
            print(f"    sent vertices       {sum(n for _, n in counts):>8,} "
                  f"(largest shape {max(n for _, n in counts)})")
            print(f"    overlay raw         {len(payload) / 1024:>8.1f} KiB")
            print(f"    overlay gzipped     {len(gzipped) / 1024:>8.1f} KiB")
            print(f"    serialise           {serialise_ms:>8.1f} ms (once per snapshot)")
            check(
                "every overlay shape is within the vertex budget",
                not over,
                f"over: {over}" if over else f"<= {settings.zones_max_vertices}",
            )
            report_timings("overlay", len(gzipped), 8.0)

            # --- hot path, with and without the opt-in geometry -------------
            centroid = wkb.loads(snapshot[0].geom_wkb).centroid
            match = index.lookup(centroid.y, centroid.x, settings.high_confidence)
            if match is None:
                print("    (centroid missed its own zone — skipping /check sizing)")
                continue

            bare = ThreatCheckResponse(
                in_risk_zone=True,
                risk_level=match.risk_level,
                threat_type=match.threat_type,
                eta_window="10–13 min",
                sector_name=match.sectors[0],
            )
            with_geom = bare.model_copy(
                update={
                    "geometry": threat_geometry_geojson(
                        index,
                        match.position,
                        settings.zones_simplify_degrees,
                        settings.zones_precision,
                        settings.zones_max_vertices,
                    )
                }
            )
            bare_bytes = len(bare.model_dump_json().encode())
            geom_bytes = len(with_geom.model_dump_json().encode())
            print(f"    /check bare         {bare_bytes:>8,} B")
            print(f"    /check +geometry    {geom_bytes:>8,} B "
                  f"({geom_bytes / max(bare_bytes, 1):.0f}x)")
            # Below GZipMiddleware's 1024-byte floor the bare response is not
            # compressed at all, which is why it is compared uncompressed.
            check(
                "bare /check stays under the 1 KiB gzip floor",
                bare_bytes < 1024,
                f"{bare_bytes} B",
            )
            report_timings(
                "/check with geometry",
                len(gzip.compress(with_geom.model_dump_json().encode(), 9))
                if geom_bytes >= 1024
                else geom_bytes,
                4.0,
            )

    await engine.dispose()

    print()
    if failures:
        print(f"FAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print("all payload budgets met")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
