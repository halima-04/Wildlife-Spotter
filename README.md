# Kenyan Wildlife Spotter — DSA 4050 Term Paper
**Halima Mahdi Mohammed · ID 670315 · USIU-Africa**

Deep learning for identifying Kenyan wildlife species from photos, with an interactive Streamlit dashboard.

## Marks checklist (what maps to what)
| Requirement | Where |
|---|---|
| a) Area + why deep learning (justify) | `report/` write-up, Section 1 |
| b) Mine dataset, present to class | `src/01_fetch_inaturalist.py`, `src/02_explore_data.py` → charts in `outputs/` |
| c) Test various ML, consider accuracy | `src/03_train_classical.py` (4 models), `src/04_train_deep.py` (4 models) → `outputs/results.json` |
| d) Interactive dashboard | `dashboard/app.py` |
| Write-up ≤ 2,000 words | `report/build_report.py` → `.docx` |
| Presentation (10 marks) | `PRESENTATION_OUTLINE.md` |

## Setup in VS Code (Windows PowerShell shown)
1. Unzip, then **File → Open Folder** on `wildlife-spotting`.
2. Open a terminal (**Ctrl+`**) and create an environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Press **Ctrl+Shift+P → Python: Select Interpreter** and choose the `.venv` one.

## Run order
```powershell
python src/01_fetch_inaturalist.py --per-species 250   # mines ~2,000 photos + GPS (10-20 min)
python src/02_explore_data.py                          # cleaning + charts
python src/03_train_classical.py                       # baselines (a few minutes)
python src/04_train_deep.py                            # deep models (CPU: ~30-60 min; use --quick to test first)
streamlit run dashboard/app.py                         # dashboard opens in the browser
python report/build_report.py                          # writes the .docx with YOUR real numbers
```
Or run steps 1-4 together: `python run_all.py` (add `--quick` for a fast trial).

## Tips
- **No GPU?** It still works on CPU. Run with `--quick` first, then the full run overnight, or use `--models resnet18 mobilenet_v2`.
- **iNaturalist blocked?** Use `python src/01b_fetch_kaggle_alternative.py` (needs a free Kaggle token; the map tab is disabled then).
- **Windows errors with workers?** Keep the default `--workers 0`.
- **Logo:** save the USIU-Africa logo as `report/usiu_logo.png` before building the report and it appears in the header.
- **Yellow highlighted text** in the report = things only you can write (your own interpretation of the results). Remove before submitting.
- Run `streamlit` from the project root. Models are saved in `models/`, charts in `outputs/`.
- The code is split into small files on purpose so you can explain each step to the lecturer. Read every script before presenting.
