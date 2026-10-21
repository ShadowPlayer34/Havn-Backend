"""Pull the Kyiv **oblast** shelter register into GeoJSON.

    python scripts/fetch_kyiv_oblast_shelters.py     # -> data/kyiv_oblast_shelters.geojson

Source: the map KODA (Kyiv Oblast State Administration) publishes at
https://koda.gov.ua/czyvilnyj-front/zahysni-sporudy-tochky-obigrivu/ukryttya/ —
a Google My Map, which exposes KML with `forcekml=1`.

KODA publishes two such maps. The one linked from the open-data page carries no
geometry at all (594 address-only placemarks that Google geocodes at render time);
this one carries a coordinate for every record, so it is the one worth having.
Covers all seven raions, and records opening hours and inclusivity per shelter.
"""

from __future__ import annotations

import json
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

MAP_ID = "133eWWMRjEzUA1_n5d2Ae28nwvrIP41U"
KML_URL = f"https://www.google.com/maps/d/kml?mid={MAP_ID}&forcekml=1"
OUT = Path("data/kyiv_oblast_shelters.geojson")
NS = {"k": "http://www.opengis.net/kml/2.2"}
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

#: KML field -> our property name. The register's own Cyrillic keys, verbatim —
#: note "Paйон" starts with a Latin "P" in the source data.
FIELD_MAP = {
    "Paйон": "raion",
    "Район": "raion",
    "Громада": "hromada",
    "Населений пункт": "settlement",
    "Тип населеного пункту": "settlement_type",
    "Адреса": "address",
    "Інклюзивне": "inclusive",
    "Графік роботи": "working_time",
    "Тип": "type",
    "wifi": "wifi",
    "id": "register_id",
}


def main() -> int:
    request = urllib.request.Request(KML_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=120) as response:
        root = ET.fromstring(response.read())

    features: list[dict] = []
    skipped = 0

    for folder in root.findall(".//k:Folder", NS):
        folder_name = folder.find("k:name", NS)
        layer = folder_name.text if folder_name is not None else None

        for mark in folder.findall(".//k:Placemark", NS):
            coordinates = mark.find(".//k:coordinates", NS)
            if coordinates is None or not (coordinates.text or "").strip():
                skipped += 1
                continue
            lon, lat, *_ = coordinates.text.strip().split(",")

            props: dict[str, object] = {"source": "koda", "layer": layer}
            name = mark.find("k:name", NS)
            props["name"] = (name.text or "").strip() if name is not None else ""
            for data in mark.findall(".//k:Data", NS):
                key = FIELD_MAP.get(data.get("name", ""))
                if not key:
                    continue
                value = data.find("k:value", NS)
                text = (value.text or "").strip() if value is not None else ""
                if text:
                    props[key] = text

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                    "properties": props,
                }
            )

    if not features:
        print("no placemarks with geometry — the map may have changed", file=sys.stderr)
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {len(features)} shelter(s) to {OUT}")
    if skipped:
        print(f"skipped {skipped} placemark(s) without coordinates")

    raions: dict[str, int] = {}
    for feature in features:
        raion = str(feature["properties"].get("raion") or "?")
        raions[raion] = raions.get(raion, 0) + 1
    for raion, count in sorted(raions.items(), key=lambda kv: -kv[1]):
        print(f"  {raion:<24} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
