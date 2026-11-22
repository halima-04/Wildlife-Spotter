"""Shared settings for the Kenyan Wildlife Spotting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"            # data/raw/<species>/*.jpg
META_CSV = ROOT / "data" / "metadata.csv"  # one row per image (species, lat, lon, date, file)
MODELS_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"
RESULTS_JSON = OUT_DIR / "results.json"

IMG_SIZE = 224
SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Species mined from iNaturalist (scientific names).
# Taxon and place IDs are looked up by name at run time.
SPECIES = {
    "elephant": "Loxodonta africana",
    "lion": "Panthera leo",
    "giraffe": "Giraffa camelopardalis",
    "zebra": "Equus quagga",
    "buffalo": "Syncerus caffer",
    "cheetah": "Acinonyx jubatus",
    "hippo": "Hippopotamus amphibius",
    "wildebeest": "Connochaetes taurinus",
}
