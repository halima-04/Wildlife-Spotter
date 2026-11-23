"""Shared helpers for steps 5 and 6: condition-aware dataset, model loading, evaluation, JSON results."""
import importlib
import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import OUT_DIR, SEED
from data import image_path
from robustness import CONDITIONS, random_condition

deep = importlib.import_module("04_train_deep")   # reuses build(), fit(), transforms, DEVICE
DEVICE = deep.DEVICE
ROBUST_JSON = OUT_DIR / "robustness.json"


class RobustDS(Dataset):
    """mode = 'train'  -> random condition with probability p (new random draw every epoch)
       mode = 'mixed'  -> a different fixed condition for each image (used for validation)
       mode = <name>   -> that one fixed condition for every image (used for testing)"""

    def __init__(self, df, c2i, tf, mode="clean", p=0.6):
        self.files = df["file"].tolist()
        self.y = df["species"].map(c2i).tolist()
        self.tf, self.mode, self.p = tf, mode, p
        self.names = list(CONDITIONS)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(image_path(self.files[i])).convert("RGB")
        if self.mode == "train":
            rng = np.random.default_rng()
            if rng.random() < self.p:
                img, _ = random_condition(img, rng)
        else:
            rng = np.random.default_rng(SEED * 1000 + i)      # same randomness for every model
            name = self.names[i % len(self.names)] if self.mode == "mixed" else self.mode
            img = CONDITIONS[name](img, rng)
        return self.tf(img), self.y[i]


def load_checkpoint(path):
    ck = torch.load(path, map_location="cpu")
    model, _ = deep.build(ck["name"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(DEVICE).eval(), ck["classes"]


@torch.no_grad()
def predict_all(model, ds, batch=64):
    model.eval()
    preds = []
    for x, _ in DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0):
        preds += model(x.to(DEVICE)).argmax(1).cpu().tolist()
    return np.array(preds)


def evaluate_conditions(model, df, c2i, conditions=None):
    """Accuracy of `model` on `df` under every fixed condition. Returns {condition: accuracy}."""
    y = np.array(df["species"].map(c2i).tolist())
    out = {}
    for name in conditions or CONDITIONS:
        pred = predict_all(model, RobustDS(df, c2i, deep.eval_tf, name))
        out[name] = float((pred == y).mean())
        print(f"    {name:12s} {out[name]:.2%}")
    return out


def update_robustness_json(model_name, cond_acc=None, region_acc=None, extra=None):
    OUT_DIR.mkdir(exist_ok=True)
    data = json.loads(ROBUST_JSON.read_text()) if ROBUST_JSON.exists() else {"conditions": {}, "regions": {}, "holdout": {}}
    data.setdefault("conditions", {}); data.setdefault("regions", {}); data.setdefault("holdout", {})
    if cond_acc is not None:
        data["conditions"][model_name] = cond_acc
    if region_acc is not None:
        data["regions"][model_name] = region_acc
    if extra:
        data["holdout"].update(extra)
    ROBUST_JSON.write_text(json.dumps(data, indent=2))
