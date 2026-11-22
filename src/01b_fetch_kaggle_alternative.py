"""Step 1 (alternative): use the Kaggle 'African Wildlife' dataset (buffalo, elephant, rhino, zebra).

Needs a free Kaggle account + API token (kaggle.json in ~/.kaggle/). Use this if iNaturalist is
blocked on your network. Produces the same folder layout as 01_fetch_inaturalist.py (no GPS map).

Usage:
    pip install kagglehub
    python src/01b_fetch_kaggle_alternative.py
"""
import shutil

import kagglehub
import pandas as pd

from config import META_CSV, RAW_DIR

path = kagglehub.dataset_download("biancaferreira/african-wildlife")
print("Downloaded to", path)

rows = []
for cls_dir in sorted(p for p in __import__("pathlib").Path(path).rglob("*") if p.is_dir()):
    imgs = list(cls_dir.glob("*.jpg"))
    if len(imgs) < 50:
        continue
    out = RAW_DIR / cls_dir.name
    out.mkdir(parents=True, exist_ok=True)
    for f in imgs:
        shutil.copy(f, out / f.name)
        rows.append({"species": cls_dir.name, "file": f"{cls_dir.name}/{f.name}"})

pd.DataFrame(rows).to_csv(META_CSV, index=False)
print(pd.DataFrame(rows)["species"].value_counts())
