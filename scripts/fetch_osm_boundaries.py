"""Download Ukrainian administrative boundaries from OSM, with Cyrillic names.

    python scripts/fetch_osm_boundaries.py oblasts raions kyiv_districts hromadas

Writes one GeoJSON per level into data/. Resumable: a level whose output file
already exists is skipped, so an interrupted run can simply be re-run.

Why OSM rather than geoBoundaries: geoBoundaries ships romanised names
("Bakhchysarai"), and the extraction model emits Cyrillic ("Бахчисарай").
Trigram similarity between the two alphabets is exactly 0.000, so a romanised
gazetteer matches nothing. OSM carries the native `name` tag.

Public Overpass mirrors are volunteer-run and frequently queue requests for
minutes. This script paces itself, retries across mirrors, and is meant to be
run once and left alone.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import osm2geojson

OUT_DIR = Path("data")

MIRRORS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
]

# `out body; >; out skel qt;` returns ways and nodes separately rather than
# inlining every coordinate per member (as `out geom` does), which is usually
# far cheaper for large multipolygon relations.
_TAIL = "out body; >; out skel qt;"

# Overpass `area(...)` / `map_to_area` scoping proved unreliable on the public
# mirrors here — it returned zero elements rather than an error. Bounding boxes
# work consistently, so we over-fetch and trim to Ukraine locally against the
# oblast polygons (see _filter_to_ukraine). Bbox order is (S, W, N, E).
UKRAINE_BBOX = "44.3,22.1,52.4,40.3"
KYIV_BBOX = "50.36,30.32,50.53,30.72"

LEVELS: dict[str, str] = {
    # 27 = 24 oblasts + Crimea + Kyiv + Sevastopol. ISO codes scope this one
    # precisely, so it needs no bbox and no trimming.
    "oblasts": f"""
        [out:json][timeout:600];
        relation["boundary"="administrative"]["admin_level"="4"]["ISO3166-2"~"^UA-"];
        {_TAIL}
    """,
    # Kyiv's city districts are admin_level 10, NOT 9 — level 9 around Kyiv is
    # the surrounding settlements (Вишгород, Бровари). This is the layer the
    # app's own users are most likely to be standing in.
    "kyiv_districts": f"""
        [out:json][timeout:300];
        relation["boundary"="administrative"]["admin_level"="10"]({KYIV_BBOX});
        {_TAIL}
    """,
    # Post-2020 raions. The bbox also catches Belarusian/Polish/Romanian raions,
    # which _filter_to_ukraine removes.
    "raions": f"""
        [out:json][timeout:900];
        relation["boundary"="administrative"]["admin_level"="6"]({UKRAINE_BBOX});
        {_TAIL}
    """,
    # Hromadas (~1470). By far the heaviest; fetch last.
    "hromadas": f"""
        [out:json][timeout:1800];
        relation["boundary"="administrative"]["admin_level"="7"]({UKRAINE_BBOX});
        {_TAIL}
    """,
}

#: Levels fetched by bbox, which therefore need trimming to Ukraine.
NEEDS_TRIM = {"raions", "hromadas"}

HEADERS = {"User-Agent": "AlertRadar-gazetteer/1.0 (one-off boundary import)"}


def fetch(query: str, label: str, attempts: int = 3) -> dict | None:
    """POST a query, rotating mirrors until one answers."""
    for attempt in range(1, attempts + 1):
        for url in MIRRORS:
            started = time.time()
            try:
                response = httpx.post(
                    url, data={"data": query}, timeout=1800, headers=HEADERS
                )
            except Exception as exc:  # noqa: BLE001 — one-off import script
                print(f"  [{label}] {url.split('/')[2]}: {type(exc).__name__} "
                      f"({time.time() - started:.0f}s)", flush=True)
                continue
            elapsed = time.time() - started
            host = url.split("/")[2]
            if response.status_code == 200:
                print(f"  [{label}] {host}: OK {len(response.content)/1e6:.1f} MB "
                      f"in {elapsed:.0f}s", flush=True)
                return response.json()
            print(f"  [{label}] {host}: HTTP {response.status_code} "
                  f"({elapsed:.0f}s)", flush=True)
            time.sleep(5)  # be a good citizen before hitting the next mirror
        if attempt < attempts:
            backoff = 30 * attempt
            print(f"  [{label}] all mirrors failed; retrying in {backoff}s", flush=True)
            time.sleep(backoff)
    return None


def _filter_to_ukraine(features: list[dict], label: str) -> list[dict]:
    """Drop features outside Ukraine, using the already-fetched oblast layer.

    Bounding-box queries cross the border: an admin_level=6 box over Ukraine
    also returns Belarusian and Polish raions. Rather than trust a name
    heuristic, test each polygon's representative point against the oblasts.
    """
    oblast_path = OUT_DIR / "osm_oblasts.geojson"
    if not oblast_path.exists():
        print(f"  [{label}] no oblast layer yet — skipping the Ukraine trim "
              f"(fetch 'oblasts' first)", flush=True)
        return features

    from shapely.geometry import shape
    from shapely.strtree import STRtree

    oblasts = json.loads(oblast_path.read_text(encoding="utf-8"))
    polygons = [shape(f["geometry"]) for f in oblasts["features"]]
    tree = STRtree(polygons)

    kept = []
    for feature in features:
        try:
            geometry = shape(feature["geometry"])
            point = geometry.representative_point()
        except Exception:  # noqa: BLE001 — malformed geometry is just dropped
            continue
        if len(tree.query(point, predicate="intersects")) > 0:
            kept.append(feature)
    print(f"  [{label}] trimmed to Ukraine: {len(kept)}/{len(features)} kept",
          flush=True)
    return kept


def to_geojson(raw: dict, label: str) -> dict:
    collection = osm2geojson.json2geojson(raw)
    features = []
    for feature in collection["features"]:
        tags = feature["properties"].get("tags", {})
        name = tags.get("name")
        if not name or feature["geometry"] is None:
            continue
        # Everything the resolver can match on. `name` is Ukrainian; the rest
        # become aliases so short forms ("Оболонь") and English spellings work.
        aliases = [
            tags[key]
            for key in ("name:uk", "official_name", "alt_name", "short_name", "name:en")
            if tags.get(key)
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "aliases": aliases,
                    "oblast": tags.get("addr:region") or tags.get("is_in:region"),
                    "admin_level": tags.get("admin_level"),
                    "layer": label,
                },
                "geometry": feature["geometry"],
            }
        )
    if label in NEEDS_TRIM:
        features = _filter_to_ukraine(features, label)
    return {"type": "FeatureCollection", "features": features}


def main(argv: list[str]) -> int:
    wanted = argv[1:] or list(LEVELS)
    unknown = [name for name in wanted if name not in LEVELS]
    if unknown:
        print(f"unknown level(s): {unknown}. Choose from {list(LEVELS)}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    failed = []
    for label in wanted:
        out_path = OUT_DIR / f"osm_{label}.geojson"
        if out_path.exists():
            print(f"[{label}] {out_path} exists — skipping", flush=True)
            continue
        print(f"[{label}] fetching…", flush=True)
        raw = fetch(LEVELS[label], label)
        if raw is None:
            print(f"[{label}] FAILED after all retries", flush=True)
            failed.append(label)
            continue
        collection = to_geojson(raw, label)
        out_path.write_text(
            json.dumps(collection, ensure_ascii=False), encoding="utf-8"
        )
        cyrillic = sum(
            1
            for f in collection["features"]
            if any("Ѐ" <= c <= "ӿ" for c in f["properties"]["name"])
        )
        print(f"[{label}] wrote {len(collection['features'])} features "
              f"({cyrillic} with Cyrillic names) -> {out_path}", flush=True)
        time.sleep(10)  # pacing between levels

    if failed:
        print(f"\nfailed levels: {failed} — re-run to retry just those", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
