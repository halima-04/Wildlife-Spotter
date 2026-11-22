"""Builds the term-paper write-up (.docx, max 2,000 words) and fills in YOUR real results.

Run AFTER training:   python report/build_report.py
- Reads outputs/results.json and data/metadata.csv, so every number in the paper matches your code.
- Anything it cannot find is shown as a yellow-highlighted [[placeholder]] to complete by hand.
- Optional: put the USIU-Africa logo at report/usiu_logo.png and it is placed in the header.
- Edit the text below freely; keep it under 2,000 words (the script prints the count).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
NAVY, TEAL = RGBColor(0x1E, 0x3A, 0x5F), RGBColor(0x0F, 0x76, 0x6E)
NAME, STUDENT_ID = "Halima Mahdi Mohammed", "670315"
OUT = ROOT / "report" / "DSA4050_Term_Paper_Wildlife_Spotting.docx"

# ------------------------------------------------------------------ pull real results
res_path, meta_path = ROOT / "outputs" / "results.json", ROOT / "data" / "metadata.csv"
res = json.loads(res_path.read_text()) if res_path.exists() else {}
meta = pd.read_csv(meta_path) if meta_path.exists() else None


def ph(label):
    return f"[[{label}]]"


def acc(model):
    return f"{res[model]['accuracy']:.1%}" if model in res else ph(f"{model} accuracy")


def f1(model):
    return f"{res[model]['macro_f1']:.3f}" if model in res else ph("F1")


models = {k: v for k, v in res.items() if k != "best"}
deep = {k: v for k, v in models.items() if v["family"] == "Deep Learning"}
classical = {k: v for k, v in models.items() if v["family"] == "Classical ML"}
best_deep = max(deep, key=lambda k: deep[k]["accuracy"]) if deep else None
best_cl = max(classical, key=lambda k: classical[k]["accuracy"]) if classical else None
n_img = f"{len(meta):,}" if meta is not None else ph("number of images")
n_cls = str(meta["species"].nunique()) if meta is not None else ph("number of species")
species_list = ", ".join(sorted(meta["species"].unique())) if meta is not None else ph("species list")
_diff = (deep[best_deep]["accuracy"] - classical[best_cl]["accuracy"]) * 100 if best_deep and best_cl else None
gap = f"{abs(_diff):.1f}" if _diff is not None else ph("gap")
DL_WINS = _diff is not None and _diff > 0

# ------------------------------------------------------------------ the write-up text
SECTIONS = [
    ("1. Introduction: why wildlife spotting, and why deep learning?", [
        "I chose wildlife species identification because Kenya's economy and identity depend on its wildlife. "
        "Tourism built on the Maasai Mara, Amboseli and Tsavo is a major source of foreign earnings, and conservancies "
        "and the Kenya Wildlife Service rely on monitoring to protect animals from poaching and human-wildlife conflict. "
        "Monitoring today uses camera traps, drones and tourist photos, which produce far more images than rangers "
        "or researchers can label by hand. A system that automatically says which animal is in a picture turns this "
        "backlog into usable data on population, movement and seasonality.",
        "Deep learning is the right tool for three reasons. First, the problem is visual and unstructured: the same "
        "lion appears in different poses, light, distance and partial occlusion by grass, and rules written by hand cannot "
        "cover that. Second, convolutional neural networks (CNNs) learn features directly from pixels, such as stripe "
        "patterns, manes and body outlines, instead of relying on features I design. Third, published camera-trap studies "
        "(Norouzzadeh et al., 2018; Tabak et al., 2019) show that CNNs identify species at a level useful for ecologists, "
        "and transfer learning lets a model pretrained on millions of everyday images work well with a few hundred photos "
        "per species, which matches what a student project can collect. To test this argument instead of just assuming it, "
        "I compared classical machine learning against deep learning on the same data.",
    ]),
    ("2. Data mining", [
        f"I mined the dataset from iNaturalist, a citizen-science platform with an open API. A Python script "
        f"(01_fetch_inaturalist.py) queries observations of {n_cls} large mammals recorded in Kenya ({species_list}). "
        "It keeps only research-grade observations, meaning at least two community members agreed on the species, and only "
        "photos with open Creative Commons licences. For each image it stores species, GPS location, date and place, "
        "which later power the map in the dashboard. Because the labels are community-verified, I did not have to label "
        "anything myself.",
        f"After downloading, 02_explore_data.py removes corrupted or tiny images and duplicates, leaving {n_img} images. "
        "I checked class balance, image sizes and monthly sighting counts, and these charts are in the presentation. "
        "I split the data once into 70% training, 15% validation and 15% test, stratified by species and saved to disk, "
        "so every model is evaluated on exactly the same unseen images. "
        "iNaturalist already blurs the coordinates of threatened species, which is important for ethics because exact "
        "locations of animals like lions or rhinos could help poachers (see Section 6).",
    ]),
    ("3. Methodology: algorithms tested", [
        "I trained two families of models and judged them mainly on test accuracy, with macro-F1 as a second check because "
        "accuracy alone can hide weak performance on small classes.",
        "Classical machine learning (03_train_classical.py). Images were resized to 96x96 and converted into a feature "
        "vector made of Histogram of Oriented Gradients (Dalal and Triggs, 2005), which captures shape and texture, "
        "plus colour histograms. On these features I trained Logistic Regression, an RBF Support Vector Machine, a Random "
        "Forest with 300 trees, and k-Nearest Neighbours. These are the baseline.",
        "Deep learning (04_train_deep.py, PyTorch). I trained (a) a small four-block CNN from scratch, and three "
        "ImageNet-pretrained networks fine-tuned for my species: (b) ResNet18 (He et al., 2016), (c) MobileNetV2 "
        "(Sandler et al., 2018), which is light enough for a field camera or phone, and (d) EfficientNet-B0 (Tan and Le, 2019). "
        "Transfer-learning models train in two stages: first only the new classification layer, then the whole network at a "
        "ten times lower learning rate. To reduce overfitting I used random crops, horizontal flips, colour jitter, "
        "dropout (in the scratch CNN), label smoothing, weight decay and a cosine learning-rate schedule. I kept the weights from the epoch "
        "with the best validation accuracy and touched the test set only once, at the end.",
    ]),
    ("4. Results", [
        f"Table 1 shows test accuracy for all models. The best classical model was {best_cl or ph('best classical model')} "
        f"at {acc(best_cl) if best_cl else ph('accuracy')}, while the best deep learning model was "
        f"{best_deep or ph('best deep model')} at {acc(best_deep) if best_deep else ph('accuracy')}, a gap of "
        f"{gap} percentage points. "
        + ("This supports the argument in Section 1: learned features beat hand-crafted ones on natural wildlife photos."
           if DL_WINS else ph("Explain honestly why deep learning did not clearly win (small dataset? easy classes?)")),
        "TABLE",
        ph("Write 3-4 sentences from YOUR confusion matrices: which species are confused (e.g. buffalo vs wildebeest, or "
           "cheetah vs lion in grass) and why, using the Grad-CAM images from the dashboard. Mention training time: classical "
           "models trained in seconds, deep models took minutes."),
    ]),
    ("5. Interactive dashboard", [
        "I built the dashboard in Streamlit (dashboard/app.py) so results can be explored rather than read from a static table. "
        "It has five tabs: an overview with key numbers and class balance; a map of Kenyan sightings that can be filtered "
        "by species; a species explorer showing sample photos; a model comparison tab with accuracy bars, learning curves "
        "and confusion matrices; and a live \"Spot an animal\" tab where anyone uploads a photo and gets the predicted species "
        "with probabilities. That last tab also draws a Grad-CAM heat map (Selvaraju et al., 2017) showing the pixels the "
        "network relied on. This helps check that the model looks at the animal rather than background cues such as "
        "a savannah scene or a vehicle, and it makes the model easier to trust and explain.",
    ]),
    ("6. Limitations and ethics", [
        "The dataset covers only a few species and mostly well-lit daytime photos taken by people, which differ from "
        "night-time infrared camera-trap images, so accuracy in the field would be lower without retraining on camera-trap "
        "data. Citizen-science photos are biased towards accessible parks and charismatic animals. The model can only "
        "predict the species it was trained on and will confidently mislabel an unknown animal, so the dashboard warns "
        "when confidence is low. On ethics, I used open-licensed data with attribution kept in the metadata, and any "
        "real deployment should avoid publishing precise locations of endangered animals.",
    ]),
    ("7. Conclusion", [
        f"This project mined a Kenyan wildlife dataset, compared {len(models) if models else ph('N')} models, and delivered "
        "an interactive tool for species identification. "
        + ("Fine-tuned deep networks outperformed classical machine learning, which justifies deep learning for this problem. "
           if DL_WINS else ph("Summarise your finding on deep learning vs classical ML. "))
        + "Future work includes adding more species and "
        "night-time images, detecting several animals per photo with an object detector such as YOLO, and deploying "
        "MobileNetV2 on a low-power camera.",
    ]),
]
REFERENCES = [
    "Dalal, N. and Triggs, B. (2005). Histograms of oriented gradients for human detection. CVPR.",
    "He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. CVPR.",
    "Norouzzadeh, M. S. et al. (2018). Automatically identifying, counting, and describing wild animals in camera-trap "
    "images with deep learning. PNAS, 115(25).",
    "Sandler, M. et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. CVPR.",
    "Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.",
    "Tabak, M. A. et al. (2019). Machine learning to classify animal species in camera trap images. Methods in Ecology "
    "and Evolution, 10(4).",
    "Tan, M. and Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. ICML.",
    "iNaturalist (2026). iNaturalist API v1. https://api.inaturalist.org/v1/docs/",
]

# ------------------------------------------------------------------ docx helpers
doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(1)
sec.top_margin = sec.bottom_margin = Inches(0.9)

normal = doc.styles["Normal"]
normal.font.name, normal.font.size = "Calibri", Pt(11)
normal.paragraph_format.space_after = Pt(6)
for name, size, color in (("Title", 22, NAVY), ("Heading 1", 14, TEAL)):
    st = doc.styles[name]
    st.font.name, st.font.size, st.font.bold, st.font.color.rgb = "Calibri", Pt(size), True, color
    st.paragraph_format.space_before, st.paragraph_format.space_after = Pt(12), Pt(4)
    st.paragraph_format.keep_with_next = True


def add_text(p, text):
    for part in re.split(r"(\[\[.*?\]\])", text):
        if part.startswith("[["):
            r = p.add_run("[" + part[2:-2] + "]")
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif part:
            p.add_run(part)


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


# header: name / ID (+ logo if supplied)
hdr = sec.header
hp = hdr.paragraphs[0]
logo = ROOT / "report" / "usiu_logo.png"
if logo.exists():
    hp.add_run().add_picture(str(logo), height=Inches(0.5))
    hp.add_run("   ")
r = hp.add_run(f"{NAME}  |  ID: {STUDENT_ID}  |  USIU-Africa  |  DSA 4050")
r.font.size, r.font.color.rgb = Pt(9), NAVY
pPr = hp._p.get_or_add_pPr()
bdr = OxmlElement("w:pBdr"); b = OxmlElement("w:bottom")
for k, v in (("val", "single"), ("sz", "8"), ("space", "4"), ("color", "0F766E")):
    b.set(qn(f"w:{k}"), v)
bdr.append(b); pPr.append(bdr)

doc.add_paragraph("Kenyan Wildlife Spotter", style="Title")
sub = doc.add_paragraph()
rr = sub.add_run("Deep learning for automatic species identification in Kenyan wildlife photos")
rr.italic, rr.font.size, rr.font.color.rgb = True, Pt(12), TEAL
meta_p = doc.add_paragraph()
meta_p.add_run("DSA 4050 Term Paper  |  BSc Data Science and Analytics, USIU-Africa  |  ").font.size = Pt(10)
meta_p.add_run(f"{NAME} ({STUDENT_ID})").font.size = Pt(10)

body_words = 0
for heading, paras in SECTIONS:
    doc.add_paragraph(heading, style="Heading 1")
    for text in paras:
        if text == "TABLE":
            cap = doc.add_paragraph()
            cr = cap.add_run("Table 1. Test-set performance (same unseen images for every model)")
            cr.bold, cr.font.size = True, Pt(10)
            rows = sorted(models.items(), key=lambda kv: -kv[1]["accuracy"]) if models else []
            t = doc.add_table(rows=1 + max(len(rows), 1), cols=4)
            t.style = "Table Grid"
            for i, h in enumerate(("Model", "Family", "Accuracy", "Macro-F1")):
                c = t.rows[0].cells[i]; c.text = ""
                run = c.paragraphs[0].add_run(h); run.bold = True; run.font.color.rgb = RGBColor(255, 255, 255)
                shade(c, "1E3A5F")
            if rows:
                for i, (k, v) in enumerate(rows, 1):
                    vals = (k, v["family"], f"{v['accuracy']:.1%}", f"{v['macro_f1']:.3f}")
                    for j, val in enumerate(vals):
                        t.rows[i].cells[j].text = val
                        if k == best_deep and j == 0:
                            t.rows[i].cells[j].paragraphs[0].runs[0].bold = True
            else:
                t.rows[1].cells[0].text = "[run the training scripts first]"
            doc.add_paragraph()
            body_words += 4 * (1 + len(rows)) if rows else 6
            continue
        p = doc.add_paragraph()
        add_text(p, text)
        body_words += len(text.split())
    # (figures are added manually in Word / see README)

doc.add_paragraph("References", style="Heading 1")
for ref in REFERENCES:
    p = doc.add_paragraph(ref)
    p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.3), Inches(-0.3)
    for r_ in p.runs:
        r_.font.size = Pt(10)

# figures (if the scripts have produced them)
figs = [("outputs/class_balance.png", "Figure 1. Images per species"),
        (f"outputs/cm_{best_deep}.png" if best_deep else "", "Figure 2. Confusion matrix of the best deep model")]
for path, cap in figs:
    if path and (ROOT / path).exists():
        doc.add_picture(str(ROOT / path), width=Inches(4.2))
        doc.paragraphs[-1].alignment = 1
        c = doc.add_paragraph(cap); c.alignment = 1
        c.runs[0].italic = True; c.runs[0].font.size = Pt(9)

doc.save(OUT)
total = body_words + sum(len(h.split()) for h, _ in SECTIONS) + 12  # + title block
print(f"Saved {OUT}\nApprox. word count (excluding references): {total}  (limit 2000)")
if total > 2000:
    print("WARNING: over the 2,000-word limit - trim some text.", file=sys.stderr)
