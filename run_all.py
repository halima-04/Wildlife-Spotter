"""Runs steps 1-4 in order. Usage: python run_all.py [--quick] [--per-species 250]"""
import argparse
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="few images + 2+2 epochs, for a fast trial")
ap.add_argument("--per-species", type=int, default=250)
a = ap.parse_args()

src = Path(__file__).parent / "src"
steps = [
    ["01_fetch_inaturalist.py", "--per-species", str(60 if a.quick else a.per_species)],
    ["02_explore_data.py"],
    ["03_train_classical.py"],
    ["04_train_deep.py"] + (["--quick"] if a.quick else []),
]
for s in steps:
    print(f"\n######## {' '.join(s)}")
    subprocess.run([sys.executable, str(src / s[0]), *s[1:]], check=True, cwd=src)
print("\nDone. Now run:  streamlit run dashboard/app.py   and   python report/build_report.py")
