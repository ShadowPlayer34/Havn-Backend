"""Load the official shelter register into PostGIS.

    python scripts/fetch_kyiv_shelters.py         # Kyiv city  (KMDA)
    python scripts/fetch_kyiv_oblast_shelters.py  # Kyiv oblast (KODA)
    python scripts/load_shelters.py --replace data/kyiv_shelters.geojson
    python scripts/load_shelters.py --replace data/kyiv_oblast_shelters.geojson

`--replace` clears only the source being loaded, so city and oblast coexist.

Only official registers are ingested. An earlier version of this script also read
OSM extracts; that path is gone deliberately. A Ukraine extract of `amenity=shelter`
is bus stops and gazebos, and `military=bunker` is WWII pillboxes — many of them
ruined, most of them in fields. Neither is somewhere to send a civilian during an
alert, and a table that mixes them with the register invites exactly that mistake.

Records not flagged `actual=1` are retired entries and are skipped.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import text

from app.bootstrap import ensure_schema
from app.config import get_settings
from app.database import OFFICIAL_SOURCES, create_engine, create_sessionmaker, session_scope

#: The register has no name field; the app needs something to put on a pin.
DEFAULT_NAME = "Укриття"

BATCH_SIZE = 1_000

#: How far outside its own stated district a record may sit before the row is thrown
#: away rather than repaired.
#:
#: A mismatch of a few kilometres is usually a stale label, not a misplaced pin: Kyiv
#: metro entrances sit on district boundaries and several are filed under the wrong
#: one. Deleting those would discard the best shelters in the city, so a point that
#: lands inside *some* known district is kept and its district corrected below.
#: Past this distance the coordinate and the address contradict each other outright —
#: the oblast map has pins 28 and 83 km from the address written on them — and there
#: is no way to tell which field is wrong, so the row cannot be trusted to send anyone
#: anywhere.
DISTRICT_TOLERANCE_M = float(os.environ.get("SHELTER_DISTRICT_TOLERANCE_M", 10_000))

#: Repair before rejection: where the point sits inside a district we have a polygon
#: for, that polygon is the better answer than the register's own label.
_CORRECT_DISTRICTS = text(
    """
    UPDATE shelters s
    SET district = sec.name
    FROM sectors sec
    WHERE s.source = :source
      AND sec.name LIKE '%район'
      AND ST_Contains(sec.geom, s.geom)
      AND s.district IS DISTINCT FROM sec.name
      AND split_part(s.district, ',', 1) || ' район' IS DISTINCT FROM sec.name
    RETURNING s.id
    """
)

#: District names are not unique in Ukraine — Дніпровський and Подільський name a
#: district of Kyiv *and* a raion 300 km away. So a row is only rejected when it is
#: far from **every** polygon bearing its district's name, not the first one joined.
#: Self-limiting too: a district with no polygon at all is never validated, so a gap
#: in the gazetteer means "unchecked", never "deleted".
_VALIDATE_AGAINST_DISTRICTS = text(
    """
    DELETE FROM shelters s
    WHERE s.source = :source
      AND s.district IS NOT NULL
      AND EXISTS (
          SELECT 1 FROM sectors sec
          WHERE sec.name = split_part(s.district, ',', 1) || ' район'
      )
      AND NOT EXISTS (
          SELECT 1 FROM sectors sec
          WHERE sec.name = split_part(s.district, ',', 1) || ' район'
            AND ST_DWithin(s.geom::geography, sec.geom::geography, :tolerance)
      )
    RETURNING s.id, s.name, s.address, s.district,
              (
                  SELECT round((min(ST_Distance(s.geom::geography, sec.geom::geography)) / 1000)::numeric, 1)
                  FROM sectors sec
                  WHERE sec.name = split_part(s.district, ',', 1) || ' район'
              ) AS km
    """
)

_UPSERT = text(
    """
    INSERT INTO shelters (
        id, name, address, shelter_type, source, district, kind, building_type,
        working_time, phone, owner, accessible, notes, geom
    )
    VALUES (
        :id, :name, :address, :shelter_type, :source, :district, :kind, :building_type,
        :working_time, :phone, :owner, :accessible, :notes,
        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)
    )
    ON CONFLICT (id) DO UPDATE SET
        name          = EXCLUDED.name,
        address       = EXCLUDED.address,
        shelter_type  = EXCLUDED.shelter_type,
        source        = EXCLUDED.source,
        district      = EXCLUDED.district,
        kind          = EXCLUDED.kind,
        building_type = EXCLUDED.building_type,
        working_time  = EXCLUDED.working_time,
        phone         = EXCLUDED.phone,
        owner         = EXCLUDED.owner,
        accessible    = EXCLUDED.accessible,
        notes         = EXCLUDED.notes,
        geom          = EXCLUDED.geom
    """
)


def features(path: Path) -> Iterator[dict]:
    """Yield GeoJSON features from a FeatureCollection or a GeoJSONSeq file."""
    raw = path.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if stripped.startswith("{") and '"FeatureCollection"' in stripped[:200]:
        yield from json.loads(raw).get("features", [])
        return
    for line in raw.splitlines():
        line = line.strip().lstrip("\x1e")
        if line:
            yield json.loads(line)


#: Register wording -> coarse category. Ordered: the first match wins.
_KMDA_TYPES = (
    ("подвійного призначення", "dual_purpose"),
    ("протирадіаційн", "radiation_shelter"),
    ("сховище", "bomb_shelter"),
    ("найпростіше", "simple_shelter"),
)


def kmda_shelter_type(props: dict) -> str:
    haystack = f"{props.get('type') or ''} {props.get('kind') or ''}".lower()
    for needle, category in _KMDA_TYPES:
        if needle in haystack:
            return category
    return "shelter"


#: KODA type -> coarse category.
_KODA_TYPES = {
    "сховище": "bomb_shelter",
    "пру": "radiation_shelter",
    "споруда подвійного призначення": "dual_purpose",
    "найпростіше укриття": "simple_shelter",
}


def koda_row(feature: dict) -> dict | None:
    """One Kyiv-oblast record -> one row."""
    props = feature.get("properties") or {}
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "Point":
        return None
    lon, lat = geometry["coordinates"][:2]

    register_id = props.get("register_id")
    if not register_id:
        return None

    address = (props.get("address") or "").strip() or None
    name = (props.get("name") or "").strip() or address or DEFAULT_NAME
    raion = (props.get("raion") or "").strip()
    hromada = (props.get("hromada") or "").strip()
    district = ", ".join(part for part in (raion, hromada) if part) or None

    shelter_type = _KODA_TYPES.get((props.get("type") or "").strip().lower(), "shelter")

    inclusive = (props.get("inclusive") or "").strip().lower()
    accessible = inclusive.startswith("так") if inclusive else None

    # "wifi: Так" is the only extra worth surfacing — it means someone can reach
    # relatives from inside, which matters over a long alert.
    notes = "Wi-Fi available" if (props.get("wifi") or "").strip().lower().startswith("так") else None

    return {
        "id": f"koda{register_id}"[:32],
        "name": name[:200],
        "address": address[:300] if address else None,
        "shelter_type": shelter_type,
        "source": "koda",
        "district": district[:120] if district else None,
        "kind": (props.get("type") or None),
        "building_type": (props.get("settlement_type") or None),
        "working_time": (props.get("working_time") or None),
        "phone": None,
        "owner": None,
        "accessible": accessible,
        "notes": notes,
        "lon": float(lon),
        "lat": float(lat),
    }


def kmda_row(feature: dict) -> dict | None:
    """One register record -> one row, or None if it is retired or unplaceable."""
    props = feature.get("properties") or {}
    if props.get("actual") != 1:
        return None

    lat, lon = props.get("lat"), props.get("long")
    if lat is None or lon is None:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") != "Point":
            return None
        lon, lat = geometry["coordinates"][:2]
    if not (-90 <= float(lat) <= 90) or not (-180 <= float(lon) <= 180):
        return None

    # The register puts the building kind in `title` and the street in `address`;
    # neither alone is a usable label on a pin.
    label = (props.get("title") or props.get("type_building") or "").strip()
    address = (props.get("address") or "").strip() or None
    name = f"{label}, {address}" if label and address else (label or address or DEFAULT_NAME)

    accessible = None
    invalid = (props.get("invalid") or "").strip().lower()
    if invalid:
        accessible = invalid.startswith("в наявності")

    return {
        "id": f"kmda{props.get('objectid')}",
        "name": name[:200],
        "address": address[:300] if address else None,
        "shelter_type": kmda_shelter_type(props),
        "source": "kmda",
        "district": (props.get("district") or None),
        "kind": (props.get("kind") or None),
        "building_type": (props.get("type_building") or None),
        "working_time": (props.get("working_time") or None),
        # `tel` is human-readable; `phonenumb` is an HTML anchor blob.
        "phone": (props.get("tel") or None),
        "owner": (props.get("owner") or None),
        "accessible": accessible,
        "notes": (props.get("description") or None),
        "lon": float(lon),
        "lat": float(lat),
    }


async def load(path: Path, replace: bool) -> tuple[int, Counter]:
    settings = get_settings()
    engine = create_engine(settings)
    sessionmaker = create_sessionmaker(engine)
    await ensure_schema(engine)

    kinds: Counter = Counter()
    skipped = 0
    batch: list[dict] = []
    total = 0

    feature_list = list(features(path))
    # Fetchers stamp `source`; the KMDA export predates that and is recognised by
    # its own primary key. Anything else is refused outright — this table answers
    # "where do I run", and may not be seeded from a source nobody vouches for.
    head = [f.get("properties") or {} for f in feature_list[:5]]
    source = next((str(p["source"]) for p in head if p.get("source")), None)
    if source is None and any("objectid" in p for p in head):
        source = "kmda"
    if source not in OFFICIAL_SOURCES:
        raise SystemExit(
            f"{path} is not a recognised official register export "
            f"(source={source!r}). Accepted: {', '.join(sorted(OFFICIAL_SOURCES))}."
        )
    adapter = koda_row if source == "koda" else kmda_row
    print(f"source: {source} ({len(feature_list)} feature(s) in {path})")

    async with session_scope(sessionmaker) as session:
        if replace:
            # Only this source's rows — the two datasets coexist in one table.
            removed = await session.execute(
                text("DELETE FROM shelters WHERE source = :source RETURNING id"),
                {"source": source},
            )
            print(f"replaced: removed {len(removed.fetchall())} existing {source} shelter(s)")

        for feature in feature_list:
            row = adapter(feature)
            if row is None:
                skipped += 1
                continue
            kinds[row["shelter_type"]] += 1
            batch.append(row)

            if len(batch) >= BATCH_SIZE:
                await session.execute(_UPSERT, batch)
                total += len(batch)
                batch.clear()

        if batch:
            await session.execute(_UPSERT, batch)
            total += len(batch)

    # Judge every row against the district it *claims*, and only then repair labels on
    # the survivors. The other order lets a pin 28 km from its own address be quietly
    # relabelled to wherever it landed, which is precisely the record worth dropping.
    async with session_scope(sessionmaker) as session:
        rejected = (
            await session.execute(
                _VALIDATE_AGAINST_DISTRICTS,
                {"source": source, "tolerance": DISTRICT_TOLERANCE_M},
            )
        ).mappings().all()
    if rejected:
        print(f"\nrejected {len(rejected)} record(s) whose coordinates contradict their address:")
        for row in sorted(rejected, key=lambda r: -float(r["km"]))[:5]:
            print(f"  {float(row['km']):>5.1f} km off  {(row['address'] or row['name'])[:58]}")
        total -= len(rejected)

    async with session_scope(sessionmaker) as session:
        corrected = (
            await session.execute(_CORRECT_DISTRICTS, {"source": source})
        ).fetchall()
    if corrected:
        print(f"corrected the district label on {len(corrected)} surviving record(s)")

    # Replacing the whole table leaves the planner working from stats for the old
    # one, which is enough to push the KNN off the index. ANALYZE outside the
    # transaction the writes ran in.
    async with session_scope(sessionmaker) as session:
        await session.execute(text("ANALYZE shelters"))

    await engine.dispose()
    if skipped:
        print(f"skipped {skipped} feature(s): retired, unplaceable, or not usable as shelter")
    return total, kinds


async def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--replace"]
    replace = "--replace" in sys.argv
    if not args:
        print(
            "usage: python scripts/load_shelters.py [--replace] <register.geojson>",
            file=sys.stderr,
        )
        return 1
    path = Path(args[0])
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    total, kinds = await load(path, replace)
    print(f"\nloaded {total} shelter(s) from {path}")
    print("by shelter_type:")
    for kind, count in kinds.most_common(10):
        print(f"  {kind:<22} {count}")

    print("\nevery row is a current entry in an official register.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
