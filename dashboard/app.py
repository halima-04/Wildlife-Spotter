"""Kenyan Wildlife Spotter - interactive dashboard (Streamlit).

Run from the project root:   streamlit run dashboard/app.py
Tabs: Overview | Sightings map | Species explorer | Model comparison | Spot an animal | Stress test (night / distance / haze)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import IMG_SIZE, META_CSV, MODELS_DIR, OUT_DIR, RAW_DIR, RESULTS_JSON  # noqa: E402

NAVY, TEAL, SAND = "#1E3A5F", "#0F766E", "#F4EFE6"
st.set_page_config(page_title="Kenyan Wildlife Spotter", page_icon="🦁", layout="wide")
st.markdown(f"""
<style>
h1, h2, h3 {{ color: {NAVY}; }}
div[data-testid="stMetric"] {{ background:{SAND}; border-left:6px solid {TEAL}; padding:12px 16px; border-radius:8px; }}
.hero {{ background:linear-gradient(90deg,{NAVY},{TEAL}); color:white; padding:22px 28px; border-radius:12px; }}
.hero h1 {{ color:white; margin:0; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data loading
@st.cache_data
def load_meta():
    return pd.read_csv(META_CSV) if META_CSV.exists() else pd.DataFrame()


@st.cache_data
def load_results():
    return json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}


meta, results = load_meta(), load_results()
if meta.empty:
    st.error("No dataset found. Run `python src/01_fetch_inaturalist.py` first (see README).")
    st.stop()

st.markdown('<div class="hero"><h1>🦁 Kenyan Wildlife Spotter</h1>'
            '<div>Deep-learning species identification from camera-trap and safari photos &nbsp;|&nbsp; '
            'DSA 4050 Term Project · USIU-Africa</div></div>', unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["📊 Overview", "🗺️ Sightings map", "🖼️ Species explorer", "🏆 Model comparison", "🔍 Spot an animal", "🌙 Stress test"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Images", f"{len(meta):,}")
    c[1].metric("Species", meta["species"].nunique())
    if results:
        best = results["best"]
        c[2].metric("Best model", best)
        c[3].metric("Test accuracy", f"{results[best]['accuracy']:.1%}")
    counts = meta["species"].value_counts().reset_index()
    counts.columns = ["species", "images"]
    left, right = st.columns(2)
    left.plotly_chart(px.bar(counts, x="species", y="images", color="species", title="Images per species",
                             color_discrete_sequence=px.colors.qualitative.Safe).update_layout(showlegend=False),
                      width="stretch")
    right.plotly_chart(px.pie(counts, names="species", values="images", hole=0.5, title="Class balance",
                              color_discrete_sequence=px.colors.qualitative.Safe), width="stretch")
    st.info("**Why deep learning?** Animals vary in pose, light, distance and background. Hand-crafted features "
            "(HOG, colour) struggle with this; CNNs learn the stripes, manes and body shapes themselves. "
            "See the *Model comparison* tab for the evidence.")

# ---------------------------------------------------------------- map
with tabs[1]:
    if "lat" in meta and meta["lat"].notna().any():
        sel = st.multiselect("Species", sorted(meta["species"].unique()), default=sorted(meta["species"].unique()))
        m = meta[meta.species.isin(sel)].dropna(subset=["lat", "lon"])
        fn = getattr(px, "scatter_map", None) or px.scatter_mapbox
        kw = {} if hasattr(px, "scatter_map") else {"mapbox_style": "open-street-map"}
        fig = fn(m, lat="lat", lon="lon", color="species", hover_name="species", hover_data=["place", "date"],
                 zoom=5, height=600, center={"lat": 0.0, "lon": 37.9}, **kw,
                 color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption("Every point is a verified (research-grade) sighting from iNaturalist, used here as training data.")
        if "date" in m and m["date"].notna().any():
            mm = pd.to_datetime(m["date"], errors="coerce").dt.month.value_counts().sort_index()
            st.plotly_chart(px.bar(x=mm.index, y=mm.values, labels={"x": "Month", "y": "Sightings"},
                                   title="When are animals photographed?", color_discrete_sequence=[TEAL]),
                            width="stretch")
    else:
        st.warning("No GPS metadata (Kaggle dataset used). Use the iNaturalist script to enable the map.")

# ---------------------------------------------------------------- explorer
with tabs[2]:
    sp = st.selectbox("Choose a species", sorted(meta["species"].unique()))
    files = meta[meta.species == sp]["file"].tolist()
    n = st.slider("How many photos?", 4, 24, 8, step=4)
    cols = st.columns(4)
    for i, f in enumerate(files[:n]):
        try:
            cols[i % 4].image(Image.open(RAW_DIR / f), width="stretch")
        except Exception:
            pass

# ---------------------------------------------------------------- model comparison
with tabs[3]:
    if not results:
        st.warning("Train the models first (steps 3 and 4 in the README).")
    else:
        rows = [{"model": k, "family": v["family"], "accuracy": v["accuracy"], "macro_f1": v["macro_f1"],
                 "train_min": v["train_seconds"] / 60} for k, v in results.items() if k != "best"]
        df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
        st.plotly_chart(px.bar(df, x="accuracy", y="model", color="family", orientation="h", text=df["accuracy"].map("{:.1%}".format),
                               title="Test accuracy by model", color_discrete_map={"Classical ML": "#94A3B8", "Deep Learning": TEAL})
                        .update_layout(yaxis={"categoryorder": "total ascending"}), width="stretch")
        st.dataframe(df.style.format({"accuracy": "{:.2%}", "macro_f1": "{:.3f}", "train_min": "{:.1f}"}), width="stretch")
        pick = st.selectbox("Inspect a model", df["model"].tolist())
        r = results[pick]
        a, b = st.columns(2)
        cm = np.array(r["confusion_matrix"])
        a.plotly_chart(px.imshow(cm, x=r["classes"], y=r["classes"], text_auto=True, color_continuous_scale="Teal",
                                 labels=dict(x="Predicted", y="Actual"), title="Confusion matrix"), width="stretch")
        if "history" in r:
            h = r["history"]
            fig = go.Figure()
            fig.add_scatter(y=h["train_acc"], name="train", line_color=NAVY)
            fig.add_scatter(y=h["val_acc"], name="validation", line_color=TEAL)
            fig.update_layout(title="Learning curve", xaxis_title="epoch", yaxis_title="accuracy")
            b.plotly_chart(fig, width="stretch")
        else:
            per = cm.diagonal() / cm.sum(1)
            b.plotly_chart(px.bar(x=r["classes"], y=per, labels={"x": "species", "y": "recall"}, title="Per-species recall",
                                  color_discrete_sequence=[TEAL]), width="stretch")


# ---------------------------------------------------------------- live prediction
def load_deep(path):
    import torch
    ck = torch.load(path, map_location="cpu")
    n, classes = ck["name"], ck["classes"]
    sys.path.insert(0, str(ROOT / "src"))
    import importlib
    mod = importlib.import_module("04_train_deep")
    model, _ = mod.build(n, len(classes), pretrained=False)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, classes, n


def last_conv_layer(model, name):
    return {"simple_cnn": lambda m: m.features, "resnet18": lambda m: m.layer4,
            "mobilenet_v2": lambda m: m.features[-1], "efficientnet_b0": lambda m: m.features[-1]}[name](model)


def gradcam(model, name, x, cls):
    import torch
    acts, grads = {}, {}
    layer = last_conv_layer(model, name)
    def fwd(m, i, o):
        acts["a"] = o
        o.register_hook(lambda g: grads.update(g=g))
    h1 = layer.register_forward_hook(fwd)
    out = model(x)
    model.zero_grad(); out[0, cls].backward()
    h1.remove()
    w = grads["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1))[0].detach().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


with tabs[4]:
    deep_files = sorted(p for p in MODELS_DIR.glob("*.pt") if p.name != "best_model.pt")
    if not deep_files:
        st.warning("No trained deep model found. Run `python src/04_train_deep.py`.")
    else:
        default = next((i for i, p in enumerate(deep_files) if results and p.stem == results.get("best")), 0)
        chosen = st.selectbox("Model", deep_files, index=default, format_func=lambda p: p.stem)
        up = st.file_uploader("Upload a wildlife photo", type=["jpg", "jpeg", "png"])
        if up:
            import torch
            from torchvision import transforms
            model, classes, mname = load_deep(chosen)
            img = Image.open(up).convert("RGB")
            tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            x = tf(img).unsqueeze(0).requires_grad_(True)
            with torch.no_grad():
                probs = torch.softmax(model(x), 1)[0].numpy()
            top = int(probs.argmax())
            c1, c2, c3 = st.columns(3)
            c1.image(img, caption="Your photo", width="stretch")
            cam = gradcam(model, mname, x, top)
            cam_img = Image.fromarray((cam * 255).astype("uint8")).resize(img.size, Image.BILINEAR)
            heat = np.asarray(cam_img) / 255.0
            overlay = (0.55 * np.asarray(img) + 0.45 * np.stack([heat * 255, heat * 120, (1 - heat) * 80], -1)).astype("uint8")
            c2.image(overlay, caption="Where the model looked (Grad-CAM)", width="stretch")
            c3.metric("Prediction", classes[top].title(), f"{probs[top]:.1%} confident")
            c3.plotly_chart(px.bar(x=probs, y=classes, orientation="h", labels={"x": "probability", "y": ""},
                                   color_discrete_sequence=[TEAL]).update_layout(height=320, yaxis={"categoryorder": "total ascending"}),
                            width="stretch")
            if probs[top] < 0.6:
                st.warning("Low confidence - the animal may be partly hidden, far away, or not one of the trained species.")


# ---------------------------------------------------------------- stress test (night, distance, surroundings)
cached_deep = st.cache_resource(load_deep)


def predict_probs(model, img):
    import torch
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        return torch.softmax(model(tf(img).unsqueeze(0)), 1)[0].numpy()


def scan_for_animals(model, classes, img, thr, max_windows=80):
    """Experimental: slide square windows of 3 sizes over the photo, classify each, keep confident ones."""
    import torch
    from PIL import ImageDraw
    from torchvision import transforms
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    w, h = img.size
    boxes = []
    for frac in (1.0, 0.5, 0.33):
        win = int(min(w, h) * frac)
        stride = max(1, win // 2)
        xs = sorted(set(list(range(0, max(1, w - win + 1), stride)) + [max(0, w - win)]))
        ys = sorted(set(list(range(0, max(1, h - win + 1), stride)) + [max(0, h - win)]))
        boxes += [(x, y, x + win, y + win) for y in ys for x in xs]
    boxes = boxes[:max_windows]
    probs = []
    with torch.no_grad():
        for i in range(0, len(boxes), 32):
            batch = torch.stack([tf(img.crop(b)) for b in boxes[i:i + 32]])
            probs += torch.softmax(model(batch), 1).numpy().tolist()
    probs = np.array(probs)
    order = np.argsort(-probs.max(1))
    kept = []
    for i in order:
        if probs[i].max() < thr:
            break
        bx = boxes[i]
        ok = True
        for k in kept:
            ix = max(0, min(bx[2], k[0][2]) - max(bx[0], k[0][0])); iy = max(0, min(bx[3], k[0][3]) - max(bx[1], k[0][1]))
            inter = ix * iy
            union = (bx[2] - bx[0]) * (bx[3] - bx[1]) + (k[0][2] - k[0][0]) * (k[0][3] - k[0][1]) - inter
            if inter / union > 0.3:
                ok = False; break
        if ok:
            kept.append((bx, classes[int(probs[i].argmax())], float(probs[i].max())))
    out = img.copy()
    d = ImageDraw.Draw(out)
    for bx, name, p in kept[:8]:
        d.rectangle(bx, outline=(255, 140, 0), width=max(2, w // 200))
        d.text((bx[0] + 4, bx[1] + 4), f"{name} {p:.0%}", fill=(255, 255, 255))
    return out, kept


with tabs[5]:
    sys.path.insert(0, str(ROOT / "src"))
    from robustness import TIMES_OF_DAY, stress  # noqa: E402  (PIL + numpy only)

    st.subheader("Stress test: can the model still spot the animal?")
    st.caption("Night, camera-trap infrared, distance and dusty haze are SIMULATED on the photo you choose. "
               "They are for testing; real night photos are harder.")
    stress_models = sorted(p for p in MODELS_DIR.glob("*.pt") if p.name != "best_model.pt")
    if not stress_models:
        st.warning("No trained deep model found. Run `python src/04_train_deep.py` (and `06_train_robust.py`).")
    else:
        src_choice = st.radio("Photo", ["Upload my own", "Random photo from the dataset"], horizontal=True)
        base_img = None
        if src_choice == "Upload my own":
            up2 = st.file_uploader("Upload a wildlife photo", type=["jpg", "jpeg", "png"], key="stress_up")
            if up2:
                base_img = Image.open(up2).convert("RGB")
        else:
            if st.button("🎲 New random photo") or "stress_pick" not in st.session_state:
                st.session_state["stress_pick"] = int(np.random.randint(0, len(meta)))
            row = meta.iloc[min(st.session_state["stress_pick"], len(meta) - 1)]
            try:
                base_img = Image.open(RAW_DIR / row["file"]).convert("RGB")
                st.caption(f"Dataset photo, true species: **{row['species']}**")
            except Exception:
                st.info("Dataset photos are not on this computer (data/raw is not in Git). Upload a photo instead.")

        c1, c2, c3 = st.columns(3)
        tod = c1.select_slider("Time of day", TIMES_OF_DAY, value="Day")
        size_pct = c2.slider("Animal size in frame (%)", 15, 100, 100, step=5, help="Lower = animal further away")
        haze_lv = c3.slider("Dust / haze", 0.0, 0.7, 0.0, step=0.05)
        default_pick = [p for p in stress_models if p.stem in ("mobilenet_v2", "mobilenet_v2_robust")] or stress_models[:1]
        chosen_models = st.multiselect("Models to compare", stress_models, default=default_pick, format_func=lambda p: p.stem)

        if base_img is not None and chosen_models:
            shown = stress(base_img, tod, size_pct / 100, haze_lv, seed=1)
            a, b = st.columns(2)
            a.image(base_img, caption="Original", width="stretch")
            b.image(shown, caption=f"{tod}, animal at {size_pct}% size, haze {haze_lv:.2f}", width="stretch")
            cols = st.columns(len(chosen_models))
            for col, path in zip(cols, chosen_models):
                model, classes, _ = cached_deep(path)
                probs = predict_probs(model, shown)
                top = int(probs.argmax())
                col.metric(path.stem, classes[top].title(), f"{probs[top]:.0%} confident")
                col.plotly_chart(px.bar(x=probs, y=classes, orientation="h", labels={"x": "probability", "y": ""},
                                        color_discrete_sequence=[TEAL]).update_layout(height=260, margin=dict(t=10, b=10),
                                        yaxis={"categoryorder": "total ascending"}), width="stretch")
            st.markdown("##### Scan for distant animals (experimental)")
            st.caption("Slides windows of three sizes over the image and keeps confident detections. "
                       "Background patches can be mislabelled, so raise the threshold if you see false boxes.")
            if st.checkbox("Run the scan on the stressed photo"):
                thr = st.slider("Confidence threshold", 0.5, 0.99, 0.9, step=0.01)
                scan_model, scan_classes, _ = cached_deep(chosen_models[-1])
                scanned, found = scan_for_animals(scan_model, scan_classes, shown, thr)
                st.image(scanned, caption=f"Scan with {chosen_models[-1].stem}", width="stretch")
                if not found:
                    st.info("No confident animal found. Lower the threshold or use a larger animal size.")

    st.divider()
    st.subheader("Measured robustness (test set)")
    rob_path = OUT_DIR / "robustness.json"
    if not rob_path.exists():
        st.info("Run `python src/05_robustness_eval.py` (in Colab) and copy `outputs/robustness.json` here to see results.")
    else:
        rob = json.loads(rob_path.read_text())
        conds = rob.get("conditions", {})
        if conds:
            rows = [{"model": m, "condition": c, "accuracy": a} for m, d in conds.items() for c, a in d.items()]
            rdf = pd.DataFrame(rows)
            st.plotly_chart(px.bar(rdf, x="condition", y="accuracy", color="model", barmode="group",
                                   title="Accuracy under simulated field conditions",
                                   color_discrete_sequence=[NAVY, TEAL, "#D97706", "#94A3B8"]).update_yaxes(tickformat=".0%", range=[0, 1]),
                            width="stretch")
        regs = rob.get("regions", {})
        if regs:
            rr = pd.DataFrame([{"model": m, "region": r, "accuracy": v["accuracy"], "photos": v["n"]}
                               for m, d in regs.items() for r, v in d.items()])
            st.plotly_chart(px.bar(rr, x="region", y="accuracy", color="model", barmode="group", hover_data=["photos"],
                                   title="Accuracy by Kenyan region (regions with 10+ test photos)",
                                   color_discrete_sequence=[NAVY, TEAL, "#D97706", "#94A3B8"]).update_yaxes(tickformat=".0%", range=[0, 1]),
                            width="stretch")
        hold = rob.get("holdout", {})
        if hold:
            st.markdown("**Unseen-area test:** model trained without photos from a region, then tested on that region.")
            st.dataframe(pd.DataFrame(hold).T.rename(columns={"unseen_region_accuracy": "unseen area", "seen_regions_accuracy": "seen areas"})
                         .style.format({"unseen area": "{:.1%}", "seen areas": "{:.1%}"}), width="stretch")
        ex = OUT_DIR / "stress_examples.png"
        if ex.exists():
            st.image(str(ex), caption="What the simulated conditions look like", width="stretch")
