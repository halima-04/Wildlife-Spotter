"""Step 3: classical ML baselines on hand-crafted features (HOG shape + colour histograms).

Models: Logistic Regression, Linear/RBF SVM, Random Forest, k-NN.
These give a baseline to show WHY deep learning is justified (see write-up, section 2).
"""
import time

import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from config import MODELS_DIR, OUT_DIR, SEED
from data import class_names, get_splits, image_path
from results_io import plot_confusion, save_result

SIZE = 96


def features(rel):
    im = Image.open(image_path(rel)).convert("RGB").resize((SIZE, SIZE))
    arr = np.asarray(im)
    gray = np.asarray(im.convert("L"))
    h = hog(gray, orientations=9, pixels_per_cell=(12, 12), cells_per_block=(2, 2))
    hist = np.concatenate([np.histogram(arr[..., c], bins=16, range=(0, 255))[0] for c in range(3)])
    return np.concatenate([h, hist / hist.sum()])


def build(df):
    X = np.array([features(f) for f in tqdm(df["file"], desc="features")])
    return X


def main():
    df = get_splits()
    classes = class_names(df)
    y_map = {c: i for i, c in enumerate(classes)}
    tr, te = df[df.split != "test"], df[df.split == "test"]   # classical models: train+val together
    Xtr, Xte = build(tr), build(te)
    ytr, yte = tr["species"].map(y_map).values, te["species"].map(y_map).values

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }
    OUT_DIR.mkdir(exist_ok=True)
    for name, clf in models.items():
        pipe = make_pipeline(StandardScaler(), clf)
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        train_s = time.time() - t0
        pred = pipe.predict(Xte)
        acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
        cm = confusion_matrix(yte, pred)
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        plot_confusion(cm, classes, f"{name} (acc {acc:.2%})", OUT_DIR / f"cm_{slug}.png")
        joblib.dump(pipe, MODELS_DIR / f"{slug}.joblib")
        save_result(name, {"family": "Classical ML", "accuracy": acc, "macro_f1": f1,
                           "train_seconds": train_s, "confusion_matrix": cm.tolist(), "classes": classes})


if __name__ == "__main__":
    main()
