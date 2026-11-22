"""Kenyan Wildlife Spotter - interactive dashboard (Streamlit).

Run from the project root:   streamlit run dashboard/app.py
Tabs: Overview | Sightings map | Species explorer | Model comparison | Spot an animal (live prediction + Grad-CAM)
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

tabs = st.tabs(["📊 Overview", "🗺️ Sightings map", "🖼️ Species explorer", "🏆 Model comparison", "🔍 Spot an animal"])

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
