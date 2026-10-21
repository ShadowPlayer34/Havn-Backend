"""Replay every sector name the model has ever emitted through the resolver.

`threat_history.raw` keeps both what the model asked for and what it got, so
the gazetteer can be re-scored against real traffic instead of invented probes
— including the names that were dropped, which exist nowhere else. Run it after
touching `normalize_sector`, `derive_aliases` or the resolution ladder:

    docker compose exec worker python scripts/audit_resolution.py

`--changed` narrows the output to names whose answer differs from the one
recorded at the time, which is the only part of a reload worth reading.
"""

from __future__ import annotations

import asyncio
import sys

from sqlalchemy import text

from app.config import get_settings
from app.database import create_engine, create_sessionmaker, session_scope
from app.geo import resolve_sectors

_PROBES = text(
    """
    SELECT raw->'extracted'->'target_sectors' AS asked,
           raw->'matched_sectors' AS matched
    FROM threat_history
    UNION ALL
    SELECT raw->'extracted'->'target_sectors', raw->'matched_sectors'
    FROM active_threats
    """
)

_AREA = text(
    "SELECT name, round((ST_Area(geom::geography) / 1e6)::numeric) AS km2 "
    "FROM sectors WHERE id = ANY(CAST(:ids AS int[]))"
)


async def main() -> int:
    only_changed = "--changed" in sys.argv
    settings = get_settings()
    engine = create_engine(settings)
    sessionmaker = create_sessionmaker(engine)

    resolved_now = 0
    still_unmatched: list[str] = []
    changed: list[tuple[str, str, str]] = []
    reports = 0

    async with session_scope(sessionmaker) as session:
        rows = (await session.execute(_PROBES)).all()
        for asked, matched in rows:
            names = list(asked or [])
            if not names:
                continue
            reports += 1
            # Resolved as a report, not name by name: the second pass uses the
            # unambiguous sectors to place the ambiguous ones.
            result = await resolve_sectors(session, names, settings.sector_match_threshold)
            before = set(matched or [])
            after = set(result.matched_names)
            areas = dict(
                (await session.execute(_AREA, {"ids": result.sector_ids})).all()
            )
            still_unmatched.extend(result.unmatched)
            resolved_now += len(after - before)
            if after != before:
                changed.append(
                    (
                        ", ".join(names),
                        ", ".join(sorted(before)) or "-",
                        ", ".join(f"{n} ({areas.get(n, '?')} km2)" for n in sorted(after))
                        or "-",
                    )
                )

    for asked, before, after in changed:
        print(f"  asked : {asked}\n  was   : {before}\n  now   : {after}\n")
    if not only_changed:
        counts: dict[str, int] = {}
        for name in still_unmatched:
            counts[name] = counts.get(name, 0) + 1
        worst = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        print("still unmatched: " + ", ".join(f"{n} x{c}" for n, c in worst))

    print(
        f"\n{reports} report(s) replayed; {len(changed)} answer(s) changed; "
        f"{resolved_now} sector(s) newly matched; {len(still_unmatched)} name(s) unmatched"
    )
    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
