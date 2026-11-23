"""Step 6: retrain a model that copes with night, infrared, distant animals, haze and golden light.

During training, 60% of images get a random simulated condition (new draw every epoch), so the model
learns to recognise species when the picture is dark, grey, grainy or the animal is small.
Validation uses a fixed mix of all conditions, so the saved weights are the ones that are best
across conditions, not just on clean photos.

Usage:
    python src/06_train_robust.py                         # mobilenet_v2 -> models/mobilenet_v2_robust.pt
    python src/06_train_robust.py --quick                 # fast smoke test
    python src/06_train_robust.py --holdout-region "Maasai Mara"
        # trains WITHOUT any Maasai Mara photos, then tests on them: how well does it handle an unseen area?
"""
import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import MODELS_DIR, OUT_DIR, SEED
from data import class_names, get_splits
from regions import assign_regions
from results_io import save_result
from robust_data import DEVICE, RobustDS, deep, evaluate_conditions, predict_all, update_robustness_json


def main(args):
    from sklearn.metrics import confusion_matrix, f1_score
    torch.manual_seed(SEED); np.random.seed(SEED)
    df = get_splits()
    classes = class_names(df)
    c2i = {c: i for i, c in enumerate(classes)}

    if args.holdout_region:
        df = df.copy()
        df["region"] = assign_regions(df["lat"].values, df["lon"].values)
        held = df[df.region == args.holdout_region]
        if len(held) < 20:
            raise SystemExit(f"Only {len(held)} photos in '{args.holdout_region}'. Regions: {sorted(df.region.unique())}")
        rest = df[df.region != args.holdout_region]
        train, val = rest[rest.split == "train"], rest[rest.split == "val"]
        test, test_in = held, rest[rest.split == "test"]
        tag = "holdout_" + args.holdout_region.lower().replace(" ", "_").replace("-", "_")
        print(f"Hold-out region '{args.holdout_region}': {len(held)} unseen photos, {len(train)} training photos")
    else:
        train, val, test = (df[df.split == s] for s in ("train", "val", "test"))
        tag = f"{args.model}_robust"
        test_in = None

    tr_dl = DataLoader(RobustDS(train, c2i, deep.train_tf, "train", p=args.p), batch_size=args.batch,
                       shuffle=True, num_workers=args.workers)
    va_dl = DataLoader(RobustDS(val, c2i, deep.eval_tf, "mixed"), batch_size=args.batch, num_workers=args.workers)

    eh, ef = (2, 2) if args.quick else (args.epochs_head, args.epochs)
    print(f"Device: {DEVICE} | model: {args.model} | epochs: {eh}+{ef}")
    model, head = deep.build(args.model, len(classes))
    model.to(DEVICE)
    t0 = time.time()
    hist = deep.fit(model, head, tr_dl, va_dl, eh, ef)
    train_s = time.time() - t0

    print("\nTest accuracy per condition:")
    cond = evaluate_conditions(model, test.reset_index(drop=True), c2i)
    MODELS_DIR.mkdir(exist_ok=True)
    torch.save({"name": args.model, "classes": classes, "state_dict": model.state_dict()}, MODELS_DIR / f"{tag}.pt")

    if args.holdout_region:
        print("\nSame model on regions it WAS trained on (test split):")
        inside = evaluate_conditions(model, test_in.reset_index(drop=True), c2i, ["clean"])["clean"]
        update_robustness_json(tag, extra={args.holdout_region: {
            "unseen_region_accuracy": cond["clean"], "seen_regions_accuracy": inside,
            "n_unseen": int(len(test)), "model": args.model}})
        print(f"\nUnseen {args.holdout_region}: {cond['clean']:.2%}  vs seen regions: {inside:.2%}")
        return

    update_robustness_json(tag, cond)
    y = np.array(test["species"].map(c2i).tolist())
    pred = predict_all(model, RobustDS(test, c2i, deep.eval_tf, "clean"))
    acc, f1 = float((pred == y).mean()), f1_score(y, pred, average="macro")
    cm = confusion_matrix(y, pred)
    (OUT_DIR / f"history_{tag}.json").write_text(json.dumps(hist))
    save_result(tag, {"family": "Deep Learning", "accuracy": acc, "macro_f1": f1, "train_seconds": train_s,
                      "params": sum(p.numel() for p in model.parameters()),
                      "confusion_matrix": cm.tolist(), "classes": classes, "history": hist})
    print(f"\nSaved models/{tag}.pt. Run  python src/05_robustness_eval.py  to compare with the baseline.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet_v2", choices=["resnet18", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.6, help="probability a training image gets a simulated condition")
    ap.add_argument("--holdout-region", default=None)
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args())
