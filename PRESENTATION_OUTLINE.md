# Presentation outline (about 10 minutes, 8 slides)

1. **Title** – Kenyan Wildlife Spotter: deep learning for species identification (name, ID, DSA 4050)
2. **The problem** – Parks and conservancies collect thousands of photos; labelling by hand is slow. Why wildlife, why Kenya.
3. **Why deep learning (justify)** – messy visuals (pose, light, grass), CNNs learn features themselves, transfer learning works with small data.
4. **Data mining** – iNaturalist API, research-grade only, open licences. Show `outputs/class_balance.png` and `outputs/sample_grid.png`. Mention cleaning and the fixed 70/15/15 split.
5. **Models tested** – Classical (HOG + colour → LogReg, SVM, RF, k-NN) vs Deep (scratch CNN, ResNet18, MobileNetV2, EfficientNet-B0).
6. **Results** – accuracy bar chart + confusion matrix of the best model (from the dashboard, *Model comparison* tab). Say which species get confused and why.
7. **Live demo** – open the dashboard: map → model comparison → upload a photo (use one the model never saw) → show Grad-CAM.
8. **Limitations, ethics, next steps** – daytime photos only, protect endangered species' locations, YOLO for multi-animal detection, MobileNetV2 on a field camera.

**Before you present:** test the demo photo beforehand, keep the dashboard already running, and have `outputs/results.json` numbers memorised (best model, accuracy, gap vs classical).
