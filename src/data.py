"""Dataset helpers: one fixed stratified split shared by ALL models so accuracy is comparable."""
import json

import pandas as pd


from config import META_CSV, MODELS_DIR, RAW_DIR, SEED, TEST_SIZE, VAL_SIZE

SPLITS_CSV = META_CSV.parent / "splits.csv"


def get_splits() -> pd.DataFrame:
    if SPLITS_CSV.exists():
        return pd.read_csv(SPLITS_CSV)
    df = pd.read_csv(META_CSV)
    from sklearn.model_selection import train_test_split
    train_val, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["species"], random_state=SEED)
    rel_val = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["species"], random_state=SEED)
    out = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    out.to_csv(SPLITS_CSV, index=False)
    return out


def class_names(df: pd.DataFrame):
    names = sorted(df["species"].unique())
    MODELS_DIR.mkdir(exist_ok=True)
    (MODELS_DIR / "classes.json").write_text(json.dumps(names))
    return names


def image_path(rel: str):
    return RAW_DIR / rel
