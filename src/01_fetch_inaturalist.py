"""Step 1 (recommended): mine wildlife photos from Kenya using the public iNaturalist API.

No account or API key is needed. Only 'research grade' observations with open licences are used.
Images go to data/raw/<species>/ and sightings (with GPS + date) go to data/metadata.csv,
which powers the map in the dashboard.

Usage:
    python src/01_fetch_inaturalist.py --per-species 250
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from config import META_CSV, RAW_DIR, SPECIES

API = "https://api.inaturalist.org/v1"
OPEN_LICENCES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
HEADERS = {"User-Agent": "USIU-DSA4050-wildlife-project/1.0 (student project)"}


def get_json(url, params):
    for attempt in range(4):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 * (attempt + 1))
    r.raise_for_status()


def find_kenya_place_id():
    data = get_json(f"{API}/places/autocomplete", {"q": "Kenya"})
    for p in data["results"]:
        if p.get("name") == "Kenya" and p.get("admin_level") == 0:
            return p["id"]
    return data["results"][0]["id"]


def find_taxon_id(sci_name):
    data = get_json(f"{API}/taxa", {"q": sci_name, "rank": "species", "per_page": 1})
    return data["results"][0]["id"]


def fetch_observations(taxon_id, place_id, wanted):
    rows, page = [], 1
    while len(rows) < wanted:
        data = get_json(
            f"{API}/observations",
            {
                "taxon_id": taxon_id, "place_id": place_id, "quality_grade": "research",
                "photo_license": OPEN_LICENCES, "photos": "true", "per_page": 100,
                "page": page, "order_by": "votes", "locale": "en",
            },
        )
        if not data["results"]:
            break
        for o in data["results"]:
            if not o.get("photos") or not o.get("location"):
                continue
            lat, lon = map(float, o["location"].split(","))
            url = o["photos"][0]["url"].replace("square", "medium")
            rows.append({"obs_id": o["id"], "lat": lat, "lon": lon,
                         "date": o.get("observed_on"), "url": url,
                         "licence": o["photos"][0].get("license_code"),
                         "place": o.get("place_guess")})
        page += 1
        time.sleep(1)  # be polite to the API
    return rows[:wanted]


def download(url, dest: Path):
    if dest.exists():
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200 and len(r.content) > 5000:
            dest.write_bytes(r.content)
            return True
    except requests.RequestException:
        pass
    return False


def main(per_species):
    place_id = find_kenya_place_id()
    print(f"Kenya place_id = {place_id}")
    all_rows = []
    for label, sci in SPECIES.items():
        folder = RAW_DIR / label
        folder.mkdir(parents=True, exist_ok=True)
        taxon = find_taxon_id(sci)
        obs = fetch_observations(taxon, place_id, per_species)
        print(f"{label}: {len(obs)} observations found")
        for o in tqdm(obs, desc=f"downloading {label}"):
            fname = f"{label}_{o['obs_id']}.jpg"
            if download(o["url"], folder / fname):
                all_rows.append({"species": label, "file": f"{label}/{fname}", **o})
    df = pd.DataFrame(all_rows)
    META_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(META_CSV, index=False)
    print(f"\nSaved {len(df)} images. Class counts:\n{df['species'].value_counts()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-species", type=int, default=250)
    main(ap.parse_args().per_species)
