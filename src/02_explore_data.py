"""Step 2: clean the dataset and explore it (use these outputs in your class presentation).

- removes corrupted / tiny images
- class balance chart, image size stats, sample grid, sightings by month
Outputs go to outputs/.
"""
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import META_CSV, OUT_DIR, RAW_DIR

OUT_DIR.mkdir(exist_ok=True)

# ---- clean ----------------------------------------------------------------
df = pd.read_csv(META_CSV)
keep, widths, heights = [], [], []
for f in df["file"]:
    try:
        with Image.open(RAW_DIR / f) as im:
            im.verify()
        with Image.open(RAW_DIR / f) as im:
            w, h = im.size
        if min(w, h) < 100:
            raise ValueError("too small")
        keep.append(True); widths.append(w); heights.append(h)
    except Exception:
        keep.append(False); widths.append(None); heights.append(None)
df["ok"], df["w"], df["h"] = keep, widths, heights
print(f"Removed {(~df['ok']).sum()} bad images")
df = df[df["ok"]].drop(columns="ok").drop_duplicates("file").reset_index(drop=True)
df.to_csv(META_CSV, index=False)

# ---- class balance ---------------------------------------------------------
counts = df["species"].value_counts()
plt.figure(figsize=(8, 4))
counts.plot.bar(color="#0F766E")
plt.title("Images per species"); plt.ylabel("count"); plt.tight_layout()
plt.savefig(OUT_DIR / "class_balance.png", dpi=150); plt.close()
print(counts)

# ---- sample grid -----------------------------------------------------------
species = sorted(df["species"].unique())
fig, axes = plt.subplots(len(species), 4, figsize=(8, 2 * len(species)))
for i, s in enumerate(species):
    sample = df[df.species == s].sample(min(4, (df.species == s).sum()), random_state=1)
    for j in range(4):
        ax = axes[i][j]; ax.axis("off")
        if j < len(sample):
            ax.imshow(Image.open(RAW_DIR / sample.iloc[j]["file"]))
        if j == 0:
            ax.set_title(s, loc="left", fontsize=9)
plt.tight_layout(); plt.savefig(OUT_DIR / "sample_grid.png", dpi=120); plt.close()

# ---- seasonality (if dates exist) ------------------------------------------
if "date" in df and df["date"].notna().any():
    m = pd.to_datetime(df["date"], errors="coerce").dt.month.value_counts().sort_index()
    plt.figure(figsize=(8, 3.5)); m.plot.bar(color="#1E3A5F")
    plt.title("Sightings by month"); plt.xlabel("month"); plt.tight_layout()
    plt.savefig(OUT_DIR / "sightings_by_month.png", dpi=150); plt.close()
print("Exploration outputs saved in outputs/")
