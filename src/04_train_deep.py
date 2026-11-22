"""Step 4: deep learning models (PyTorch).

1. Simple CNN trained from scratch
2. ResNet18 transfer learning (ImageNet weights)
3. MobileNetV2 transfer learning (light enough for a phone / field camera)
4. EfficientNet-B0 transfer learning

Transfer models train in two stages: head only, then full fine-tune at a lower learning rate.
The model with the best test accuracy is saved as models/best_model.pt for the dashboard.

Usage:
    python src/04_train_deep.py                 # all models
    python src/04_train_deep.py --quick         # 2+2 epochs, for a fast smoke test
    python src/04_train_deep.py --models resnet18 mobilenet_v2 --epochs 8
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from config import IMG_SIZE, MODELS_DIR, OUT_DIR, SEED
from data import class_names, get_splits, image_path
from results_io import plot_confusion, save_result

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


class WildlifeDS(Dataset):
    def __init__(self, df, class_to_idx, tf):
        self.files, self.tf = df["file"].tolist(), tf
        self.y = df["species"].map(class_to_idx).tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(image_path(self.files[i])).convert("RGB")
        return self.tf(img), self.y[i]


class SimpleCNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, n))

    def forward(self, x):
        return self.head(self.features(x))


def build(name, n, pretrained=True):
    if name == "simple_cnn":
        return SimpleCNN(n), None
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n)
        return m, "fc"
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.last_channel, n)
        return m, "classifier"
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n)
        return m, "classifier"
    raise ValueError(name)


def run_epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    tot_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, correct / n


def fit(model, head_name, tr_dl, va_dl, epochs_head, epochs_full):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    hist = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    best_val, best_state = 0, None

    def go(epochs, lr, params):
        nonlocal best_val, best_state
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        for ep in range(epochs):
            tl, ta = run_epoch(model, tr_dl, loss_fn, opt)
            vl, va = run_epoch(model, va_dl, loss_fn)
            sched.step()
            for k, v in zip(hist, (ta, va, tl, vl)):
                hist[k].append(v)
            if va > best_val:
                best_val, best_state = va, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  epoch {len(hist['val_acc']):02d}  train_acc={ta:.3f}  val_acc={va:.3f}  val_loss={vl:.3f}")

    if head_name:  # transfer learning: stage 1 = head only, stage 2 = everything
        for p in model.parameters():
            p.requires_grad = False
        for p in getattr(model, head_name).parameters():
            p.requires_grad = True
        go(epochs_head, 1e-3, [p for p in model.parameters() if p.requires_grad])
        for p in model.parameters():
            p.requires_grad = True
        go(epochs_full, 1e-4, model.parameters())
    else:          # from scratch
        go(epochs_head + epochs_full, 2e-3, model.parameters())
    model.load_state_dict(best_state)
    return hist


def main(args):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    torch.manual_seed(SEED); np.random.seed(SEED)
    df = get_splits()
    classes = class_names(df)
    c2i = {c: i for i, c in enumerate(classes)}
    loaders = {
        s: DataLoader(WildlifeDS(df[df.split == s], c2i, train_tf if s == "train" else eval_tf),
                      batch_size=args.batch, shuffle=(s == "train"), num_workers=args.workers)
        for s in ("train", "val", "test")
    }
    print(f"Device: {DEVICE} | classes: {classes}")
    eh, ef = (2, 2) if args.quick else (args.epochs_head, args.epochs)
    OUT_DIR.mkdir(exist_ok=True)

    for name in args.models:
        print(f"\n=== {name} ===")
        model, head = build(name, len(classes))
        model.to(DEVICE)
        t0 = time.time()
        hist = fit(model, head, loaders["train"], loaders["val"], eh, ef)
        train_s = time.time() - t0

        model.eval(); preds, ys = [], []
        with torch.no_grad():
            for x, y in loaders["test"]:
                preds += model(x.to(DEVICE)).argmax(1).cpu().tolist(); ys += y.tolist()
        acc, f1 = accuracy_score(ys, preds), f1_score(ys, preds, average="macro")
        cm = confusion_matrix(ys, preds)
        print(classification_report(ys, preds, target_names=classes))
        plot_confusion(cm, classes, f"{name} (acc {acc:.2%})", OUT_DIR / f"cm_{name}.png")
        torch.save({"name": name, "classes": classes, "state_dict": model.state_dict()}, MODELS_DIR / f"{name}.pt")
        (OUT_DIR / f"history_{name}.json").write_text(json.dumps(hist))
        save_result(name, {"family": "Deep Learning", "accuracy": acc, "macro_f1": f1,
                           "train_seconds": train_s, "params": sum(p.numel() for p in model.parameters()),
                           "confusion_matrix": cm.tolist(), "classes": classes, "history": hist})

    # copy best deep model so the dashboard can find it
    res = json.loads((OUT_DIR / "results.json").read_text())
    deep = {k: v for k, v in res.items() if k != "best" and v["family"] == "Deep Learning"}
    best = max(deep, key=lambda k: deep[k]["accuracy"])
    (MODELS_DIR / "best_model.pt").write_bytes((MODELS_DIR / f"{best}.pt").read_bytes())
    print(f"\nBest deep model: {best} ({deep[best]['accuracy']:.2%}) -> models/best_model.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["simple_cnn", "resnet18", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=6, help="full fine-tune epochs (scratch CNN gets head+full)")
    ap.add_argument("--epochs-head", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0, help="0 is safest on Windows")
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args())
