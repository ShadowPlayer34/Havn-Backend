"""Load the administrative-district gazetteer from GeoJSON into PostGIS.

    python scripts/load_sectors.py data/sectors.sample.geojson

Expected feature properties:

    name    (required) — canonical display name, e.g. "Шевченківський район"
    aliases (optional) — list of alternative spellings the channels use
    oblast  (optional) — parent region, for display only

Resolution quality is entirely a function of this table. For production, load
real boundaries — Ukrainian hromada/raion polygons are published as open data,
and OSM extracts carry `admin_level` 4/6/7 relations.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from sqlalchemy import text

from app.bootstrap import ensure_schema
from app.config import get_settings
from app.database import create_engine, create_sessionmaker, session_scope
from app.schemas import normalize_sector

# Ukrainian adjectival endings. A gazetteer stores "Харківська область" and
# "Оболонський район", but channels write the colloquial "Харківщина" and
# "Оболонь" — which score 0.375 and 0.429 against the full names, both under the
# 0.45 trigram threshold. Against the stem they score 0.500 and 0.667. So we
# derive stems at load time and store them as aliases.
_ADJECTIVAL_SUFFIXES = (
    "ського", "ської", "ський", "ська", "ське",
    "цький", "цька", "цьке",
    "зький", "зька", "зьке",
    "ький", "ька", "ьке",
    # Linking morphemes. Hromada names are built off the settlement with an
    # interfix the plain adjectival endings leave behind: "Шполянська" stops at
    # "шполян", which scores 0.444 against the bare "Шпола" — under the 0.45
    # threshold by six thousandths. Stripping the interfix as well reaches
    # "шпол" (0.571) and "Городищенська" -> "городищ" (an exact hit for
    # "Городище"). Every suffix that matches contributes a stem, so these add
    # candidates rather than replacing the ones above.
    "янського", "янської", "янський", "янська", "янське",
    "инського", "инської", "инський", "инська",
    "енського", "енської", "енський", "енська",
    "івського", "івської", "івський", "івська",
    "овського", "овської", "овський", "овська",
    "а", "и", "е",
)
_MIN_STEM = 3  # "Сумська" -> "сум" is only three characters and still valid


def derive_aliases(name_norm: str) -> set[str]:
    """Colloquial and short forms a channel might use for this sector.

    Generates several candidate stems rather than one: Ukrainian adjectival
    formation is irregular ("Харківська"->"Харків" strips four characters,
    "Одеська"->"Одес" strips three), and extra aliases cost nothing at match
    time while a missing one costs a dropped threat.
    """
    aliases: set[str] = set()
    for suffix in _ADJECTIVAL_SUFFIXES:
        if name_norm.endswith(suffix):
            stem = name_norm[: -len(suffix)]
            if len(stem) >= _MIN_STEM:
                aliases.add(stem)
                # "Харків" + "щина" = "Харківщина" — an exact hit rather than a
                # fuzzy one, which is both faster and safer.
                aliases.add(stem + "щина")
    aliases.discard(name_norm)
    return aliases

_INSERT = text(
    """
    INSERT INTO sectors (name, name_norm, aliases, oblast, geom, centroid)
    VALUES (
        :name,
        :name_norm,
        CAST(:aliases AS text[]),
        :oblast,
        ST_Multi(ST_GeomFromWKB(:geom, 4326)),
        ST_Centroid(ST_GeomFromWKB(:geom, 4326))
    )
    """
)


CURATED_ALIASES_PATH = Path("data/aliases.json")


def load_curated_aliases() -> dict[str, list[str]]:
    """Hand-maintained aliases, keyed by normalised sector name."""
    if not CURATED_ALIASES_PATH.exists():
        return {}
    raw = json.loads(CURATED_ALIASES_PATH.read_text(encoding="utf-8"))
    return {
        normalize_sector(key): [normalize_sector(v) for v in values]
        for key, values in raw.items()
        if not key.startswith("_") and isinstance(values, list)
    }


async def load(paths: list[Path], replace: bool) -> int:
    settings = get_settings()
    engine = create_engine(settings)
    sessionmaker = create_sessionmaker(engine)
    await ensure_schema(engine)

    curated = load_curated_aliases()
    if curated:
        print(f"curated aliases: {len(curated)} sector(s) from {CURATED_ALIASES_PATH}")
    curated_used: set[str] = set()

    total = 0
    async with session_scope(sessionmaker) as session:
        if replace:
            removed = await session.execute(text("DELETE FROM sectors RETURNING id"))
            print(f"replaced: removed {len(removed.fetchall())} existing sector(s)")

        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            features = payload.get("features", [])
            loaded = 0
            seen: dict[str, int] = {}

            for feature in features:
                props = feature.get("properties", {})
                name = props.get("name")
                if not name:
                    print(f"  skipping feature without a name: {props}", file=sys.stderr)
                    continue

                geometry: BaseGeometry = shape(feature["geometry"])
                if geometry.is_empty:
                    print(f"  skipping empty geometry for {name}", file=sys.stderr)
                    continue

                name_norm = normalize_sector(name)
                if name_norm in curated:
                    curated_used.add(name_norm)
                aliases = sorted(
                    (
                        {normalize_sector(a) for a in props.get("aliases", []) if a}
                        | derive_aliases(name_norm)
                        | set(curated.get(name_norm, ()))
                    )
                    - {name_norm, ""}
                )
                seen[name_norm] = seen.get(name_norm, 0) + 1

                await session.execute(
                    _INSERT,
                    {
                        "name": name,
                        "name_norm": name_norm,
                        "aliases": aliases,
                        "oblast": props.get("oblast"),
                        "geom": geometry.wkb,
                    },
                )
                loaded += 1

            # Different administrative units legitimately share a name (several
            # oblasts contain a "Олександрівська" hromada). They are all kept —
            # the resolver breaks the tie by area — but it is worth knowing how
            # much ambiguity a layer introduces.
            ambiguous = {n: c for n, c in seen.items() if c > 1}
            print(f"loaded {loaded} sector(s) from {path}")
            if ambiguous:
                worst = sorted(ambiguous.items(), key=lambda kv: -kv[1])[:5]
                print(
                    f"  note: {len(ambiguous)} name(s) are not unique in this layer; "
                    f"most duplicated: {worst}"
                )
            total += loaded

        # A curated key that matched nothing is almost always a typo or a name
        # that changed — silence there would just look like the alias working.
        unused = sorted(set(curated) - curated_used)
        if unused:
            print(f"\nWARNING: {len(unused)} curated alias key(s) matched no sector: {unused}")

    await engine.dispose()
    return total


async def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--replace"]
    replace = "--replace" in sys.argv
    if not args:
        print(
            "usage: python scripts/load_sectors.py [--replace] <file.geojson> [more.geojson ...]\n"
            "  --replace  wipe the sectors table first (use when loading a full set)",
            file=sys.stderr,
        )
        return 1
    paths = [Path(a) for a in args]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"no such file(s): {missing}", file=sys.stderr)
        return 1
    count = await load(paths, replace)
    print(f"\ntotal: {count} sector(s) in the gazetteer")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
