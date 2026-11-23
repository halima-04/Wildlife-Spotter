"""Step 5: robustness tests - how does accuracy change at night, far away, in haze, and by region?

For every model checkpoint chosen, the SAME test images are re-scored under 8 conditions
(clean, night, infrared, far, very far, far+night, haze, golden hour) and per Kenyan region.
Outputs: outputs/robustness.json, outputs/robustness.png, outputs/stress_examples.png

Usage:
    python src/05_robustness_eval.py                                   # mobilenet_v2 (+ _robust if it exists)
    python src/05_robustness_eval.py --models mobilenet_v2 resnet18
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import MODELS_DIR, OUT_DIR, SEED
from data import class_names, get_splits, image_path
from regions import assign_regions
from robust_data import deep, evaluate_conditions, load_checkpoint, predict_all, update_robustness_json, RobustDS
from robustness import CONDITIONS, LABELS


def region_accuracy(model, test, c2i, min_n=10):
    if "lat" not in test or test["lat"].isna().all():
        return None
    reg = assign_regions(test["lat"].values, test["lon"].values)
    pred = predict_all(model, RobustDS(test, c2i, deep.eval_tf, "clean"))
    y = np.array(test["species"].map(c2i).tolist())
    out = {}
    for r in sorted(set(reg)):
        m = reg == r
        if m.sum() >= min_n:
            out[r] = {"accuracy": float((pred[m] == y[m]).mean()), "n": int(m.sum())}
    return out


def example_grid(test):
    row = test.sample(1, random_state=SEED).iloc[0]
    img = Image.open(image_path(row["file"])).convert("RGB")
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.2))
    for ax, name in zip(axes.ravel(), CONDITIONS):
        ax.imshow(CONDITIONS[name](img, np.random.default_rng(SEED)))
        ax.set_title(LABELS[name], fontsize=10); ax.axis("off")
    fig.suptitle(f"Simulated field conditions (example: {row['species']})")
    plt.tight_layout(); plt.savefig(OUT_DIR / "stress_examples.png", dpi=140); plt.close()


def main(args):
    df = get_splits()
    classes = class_names(df)
    c2i = {c: i for i, c in enumerate(classes)}
    test = df[df.split == "test"].reset_index(drop=True)
    example_grid(test)

    names = args.models or [n for n in ("mobilenet_v2", "mobilenet_v2_robust") if (MODELS_DIR / f"{n}.pt").exists()]
    if not names:
        names = [p.stem for p in MODELS_DIR.glob("*.pt") if p.stem != "best_model"][:2]
    print("Models to test:", names)
    for n in names:
        print(f"\n=== {n} ===")
        model, cls = load_checkpoint(MODELS_DIR / f"{n}.pt")
        assert cls == classes, "class order differs - re-run step 4/6 on the current split"
        cond = evaluate_conditions(model, test, c2i)
        update_robustness_json(n, cond, region_accuracy(model, test, c2i))

    import json
    res = json.loads((OUT_DIR / "robustness.json").read_text())["conditions"]
    conds = list(CONDITIONS)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.8 / max(len(res), 1)
    for i, (n, accs) in enumerate(res.items()):
        ax.bar(np.arange(len(conds)) + i * width, [accs.get(c, 0) * 100 for c in conds], width, label=n)
    ax.set_xticks(np.arange(len(conds)) + width * (len(res) - 1) / 2)
    ax.set_xticklabels([LABELS[c] for c in conds], rotation=20)
    ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(0, 100); ax.legend()
    ax.set_title("Accuracy under simulated field conditions")
    plt.tight_layout(); plt.savefig(OUT_DIR / "robustness.png", dpi=150); plt.close()
    print("\nSaved outputs/robustness.json, robustness.png and stress_examples.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None)
    main(ap.parse_args())
