"""Pull the official Kyiv shelter registry into GeoJSON.

    python scripts/fetch_kyiv_shelters.py            # -> data/kyiv_shelters.geojson

Source: the ArcGIS layer behind https://gis.kyivcity.gov.ua/shelter/, which KMDA
keeps in sync with the statutory register of protective structures. Unlike an OSM
extract this carries the things that decide whether a shelter is usable at 3am —
address, opening hours, owner, phone, wheelchair access — and an `actual` flag for
records that have been retired.

The service caps a response at `maxRecordCount`, so this pages with resultOffset
until it stops returning features.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

LAYER = (
    "https://gisserver.kyivcity.gov.ua/mayno/rest/services/public_protection/"
    "%D0%A3%D0%BA%D1%80%D0%B8%D1%82%D1%82%D1%8F_%D1%80%D0%B5%D0%B4%D0%B0%D0%BA%D1%82%D0%BE%D1%80"
    "/MapServer/0"
)
OUT = Path("data/kyiv_shelters.geojson")
PAGE = 1_000
# The service 403s the default urllib agent.
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def query(offset: int) -> dict:
    params = urllib.parse.urlencode(
        {
            "where": "1=1",
            "outFields": "*",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": PAGE,
        }
    )
    return get(f"{LAYER}/query?{params}")


def main() -> int:
    total = get(f"{LAYER}/query?where=1%3D1&returnCountOnly=true&f=json").get("count", 0)
    print(f"registry reports {total} record(s)")

    features: list[dict] = []
    offset = 0
    while True:
        page = query(offset)
        batch = page.get("features", [])
        if not batch:
            break
        features.extend(batch)
        print(f"  fetched {len(features)}/{total}")
        if len(batch) < PAGE:
            break
        offset += PAGE

    if not features:
        print("no features returned — the service may have changed", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {len(features)} shelter(s) to {OUT}")

    retired = sum(1 for f in features if (f.get("properties") or {}).get("actual") != 1)
    if retired:
        print(f"note: {retired} record(s) are not flagged actual=1 and will be skipped at load")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
