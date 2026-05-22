import os
import glob
import subprocess
import sys
import time
from typing import List, Tuple

from tqdm import tqdm

import argparse


# ===================== PATH SETUP ===================== #
# This file is: <project_root>/src/Pipe/run_tracking_pipeline.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

RAW_ROOT = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "proccessed")

TRACKING_SCRIPT = os.path.join(PROJECT_ROOT, "src", "Scripts", "track_ball.py")
PYTHON_EXE = sys.executable
# ====================================================== #

VIDEO_GLOB = "*.mp4"          # all MP4s in the animal folder
RECURSIVE_IN_ANIMAL = False   # set True if mp4s might be nested under animal folders
DRY_RUN = False               # True = print commands only, don't run


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_subdirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ])


def is_reports_folder(name: str) -> bool:
    return name.strip().lower() == "reports"


def iter_video_files(animal_dir: str) -> List[str]:
    if RECURSIVE_IN_ANIMAL:
        return sorted(glob.glob(os.path.join(animal_dir, "**", VIDEO_GLOB), recursive=True))
    return sorted(glob.glob(os.path.join(animal_dir, VIDEO_GLOB)))


def build_jobs(cohort_filter: str | None = None) -> List[Tuple[str, str, str, str, str]]:

    """
    Expected raw structure:
      data/raw/<cohort>/<date>/<animal>/*.mp4

    Returns list of jobs:
      (cohort_name, date_name, animal_name, video_path, out_dir)
    """
    jobs: List[Tuple[str, str, str, str, str]] = []

    cohort_dirs = list_subdirs(RAW_ROOT)
    if cohort_filter is not None:
        cohort_dirs = [c for c in cohort_dirs if c == cohort_filter]

    if not cohort_dirs:
        print(f"No matching cohort found for: {cohort_filter}")
        return []

    for cohort in cohort_dirs:
        cohort_path = os.path.join(RAW_ROOT, cohort)

        date_dirs = list_subdirs(cohort_path)
        for date in date_dirs:
            date_path = os.path.join(cohort_path, date)

            animal_dirs = list_subdirs(date_path)
            for animal in animal_dirs:
                if is_reports_folder(animal):
                    continue  # ignore "Reports" folder inside dates

                animal_path = os.path.join(date_path, animal)

                vids = iter_video_files(animal_path)
                if not vids:
                    continue

                # Output preserves cohort/date/animal:
                # data/proccessed/<cohort>/<date>/<animal>/
                out_dir = os.path.join(OUTPUT_ROOT, cohort, date, animal)
                safe_mkdir(out_dir)

                for v in vids:
                    jobs.append((cohort, date, animal, v, out_dir))

    return jobs


def main():
    parser = argparse.ArgumentParser(description="Run ball tracking pipeline")
    parser.add_argument("--cohort", type=str, default=None,
                        help="Only process this cohort (folder name under data/raw)")
    args = parser.parse_args()

    if not os.path.isdir(RAW_ROOT):
        print(f"RAW_ROOT does not exist or is not a directory: {RAW_ROOT}")
        return

    if not os.path.isfile(TRACKING_SCRIPT):
        print(f"Tracking script not found: {TRACKING_SCRIPT}")
        print("Fix: ensure track_ball.py exists at src/Scripts/track_ball.py")
        return

    safe_mkdir(OUTPUT_ROOT)

    jobs = build_jobs(cohort_filter=args.cohort)
    if not jobs:
        print(f"No videos found matching '{VIDEO_GLOB}' under: {RAW_ROOT}")
        print("Expected structure: data/raw/<Cohort X>/<DATE>/<Het ###>/*.mp4")
        return

    cohorts = sorted(set(j[0] for j in jobs))
    dates = sorted(set((j[0], j[1]) for j in jobs))
    animals = sorted(set((j[0], j[1], j[2]) for j in jobs))

    print(f"Found {len(jobs)} video(s).")
    print(f"Cohorts found: {', '.join(cohorts)}")
    print(f"Unique (cohort,date) folders: {len(dates)}")
    print(f"Unique (cohort,date,animal) folders: {len(animals)}")
    print(f"RAW_ROOT:     {RAW_ROOT}")
    print(f"OUTPUT_ROOT:  {OUTPUT_ROOT}")
    print(f"PYTHON_EXE:   {PYTHON_EXE}")
    print(f"TRACK_SCRIPT: {TRACKING_SCRIPT}")
    print("-" * 80)

    n_ok = 0
    n_fail = 0
    failures = []

    start_time = time.time()

    with tqdm(total=len(jobs), desc="Tracking videos", unit="video") as pbar:
        for idx, (cohort, date, animal, video_path, out_dir) in enumerate(jobs, start=1):
            video_name = os.path.basename(video_path)

            # Show current item
            pbar.set_postfix({
                "cohort": cohort,
                "date": date,
                "animal": animal,
                "video": video_name
            })

            cmd = [
                PYTHON_EXE,
                TRACKING_SCRIPT,
                "--video", video_path,
                "--outdir", out_dir,
            ]

            if DRY_RUN:
                print("\nDRY_RUN:", " ".join(cmd))
                pbar.update(1)
                continue

            try:
                subprocess.run(cmd, check=True)
                n_ok += 1
            except subprocess.CalledProcessError as e:
                n_fail += 1
                failures.append(video_path)
                print(f"\n❌ Failed: {video_path}\n   {e}")

            # ETA based on average
            elapsed = time.time() - start_time
            avg_per = elapsed / idx
            remaining = avg_per * (len(jobs) - idx)
            pbar.set_postfix_str(
                f"{cohort}/{date}/{animal}/{video_name} | ETA ~ {remaining/60:.1f} min"
            )

            pbar.update(1)

    print("\n" + "=" * 80)
    print("Pipeline complete")
    print(f"  Success: {n_ok}")
    print(f"  Failed:  {n_fail}")
    print(f"  Output:  {OUTPUT_ROOT}")

    if failures:
        fail_log = os.path.join(OUTPUT_ROOT, "tracking_failures.txt")
        with open(fail_log, "w", encoding="utf-8") as f:
            for v in failures:
                f.write(v + "\n")
        print(f"\nWrote failure list to: {fail_log}")

    print("=" * 80)


if __name__ == "__main__":
    main()
