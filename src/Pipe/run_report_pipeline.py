import os
import glob
import subprocess
import sys
import time
from typing import List

from tqdm import tqdm

import argparse



# ===================== PATH SETUP ===================== #
# This file is: <project_root>/src/Pipe/run_report_pipeline.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

PROCESSED_ROOT = os.path.join(PROJECT_ROOT, "data", "proccessed")
REPORTS_ROOT = os.path.join(PROJECT_ROOT, "data", "reports")

CSV_GLOB = "**/*_ball_speed_fast.csv"

REPORT_SCRIPT = os.path.join(PROJECT_ROOT, "src", "Scripts", "report_from_csv.py")
PYTHON_EXE = sys.executable
# ====================================================== #

THRESH_CM_S = 0.5
SMOOTH_WINDOW_SEC = 0.5

DRY_RUN = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_csvs(cohort_filter: str | None = None) -> List[str]:
    if cohort_filter is None:
        pattern = os.path.join(PROCESSED_ROOT, CSV_GLOB)
    else:
        pattern = os.path.join(PROCESSED_ROOT, cohort_filter, CSV_GLOB)

    return sorted(glob.glob(pattern, recursive=True))



def main():
    parser = argparse.ArgumentParser(description="Run report generation pipeline")
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Only process this cohort (folder name under data/proccessed)"
    )
    args = parser.parse_args()

    if not os.path.isdir(PROCESSED_ROOT):
        print(f"Processed root not found: {PROCESSED_ROOT}")
        return

    if not os.path.isfile(REPORT_SCRIPT):
        print(f"Report script not found: {REPORT_SCRIPT}")
        print("Expected: src/Scripts/report_from_csv.py")
        return

    ensure_dir(REPORTS_ROOT)

    csvs = find_csvs(cohort_filter=args.cohort)
    if not csvs:
        print(f"No CSVs found under: {PROCESSED_ROOT}")
        print(f"Pattern: {CSV_GLOB}")
        return

    print(f"Found {len(csvs)} CSV(s). Writing reports into: {REPORTS_ROOT}")
    print(f"Using: {REPORT_SCRIPT}")
    print("-" * 80)

    ok = 0
    fail = 0
    failures: List[str] = []

    start = time.time()

    with tqdm(total=len(csvs), desc="Generating reports", unit="csv") as pbar:
        for i, csv_path in enumerate(csvs, start=1):
            base = os.path.basename(csv_path)
            pbar.set_postfix({"csv": base})

            cmd = [
                PYTHON_EXE,
                REPORT_SCRIPT,
                "--csv", csv_path,
                "--processed-root", PROCESSED_ROOT,
                "--reports-root", REPORTS_ROOT,
                "--thresh", str(THRESH_CM_S),
                "--smooth-sec", str(SMOOTH_WINDOW_SEC),
            ]

            if DRY_RUN:
                print("\nDRY_RUN:", cmd)
                ok += 1
                pbar.update(1)
                continue

            try:
                subprocess.run(cmd, check=True)
                ok += 1
            except subprocess.CalledProcessError as e:
                fail += 1
                failures.append(csv_path)
                print(f"\n❌ Failed on {csv_path}\n   {e}")

            elapsed = time.time() - start
            avg = elapsed / i
            remaining = avg * (len(csvs) - i)
            pbar.set_postfix_str(f"{base} | ETA ~ {remaining/60:.1f} min")
            pbar.update(1)

    print("\n" + "=" * 80)
    print("Report pipeline complete")
    print(f"  Success: {ok}")
    print(f"  Failed:  {fail}")
    print(f"  Reports: {REPORTS_ROOT}")

    if failures:
        fail_log = os.path.join(REPORTS_ROOT, "report_failures.txt")
        with open(fail_log, "w", encoding="utf-8") as f:
            for p in failures:
                f.write(p + "\n")
        print(f"\nWrote failure list to: {fail_log}")

    print("=" * 80)


if __name__ == "__main__":
    main()
