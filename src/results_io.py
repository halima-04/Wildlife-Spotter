"""Tiny helper so every training script appends to the same outputs/results.json."""
import json

from config import OUT_DIR, RESULTS_JSON


def save_result(name: str, payload: dict):
    OUT_DIR.mkdir(exist_ok=True)
    data = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    data[name] = payload
    best = max((k for k in data if k != "best"), key=lambda k: data[k]["accuracy"])
    data["best"] = best
    RESULTS_JSON.write_text(json.dumps(data, indent=2))
    print(f"[saved] {name}: accuracy={payload['accuracy']:.4f}  macro-F1={payload['macro_f1']:.4f}")


def plot_confusion(cm, classes, title, path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
