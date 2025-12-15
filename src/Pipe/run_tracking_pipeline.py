import os
import glob
import subprocess
import sys
import time
from tqdm import tqdm

# ===================== USER SETTINGS ===================== #
RAW_ROOT = r"C:\Users\16083\Coding\Behaviour_Tracking_Self\data\raw"
OUTPUT_ROOT = r"C:\Users\16083\Coding\Behaviour_Tracking_Self\data\proccessed"

TRACKING_SCRIPT = r"C:\Users\16083\Coding\Behaviour_Tracking_Self\src\Scripts\track_ball.py"
PYTHON_EXE = sys.executable  # uses current venv python

VIDEO_GLOB = "post*.mp4"     # match within each animal folder
DRY_RUN = False             # True = print commands only, don't run
# ========================================================= #


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def animal_output_folder(animal_dir_name: str) -> str:
    # You asked for folders like: "proccessed_het230_post"
    return os.path.join(OUTPUT_ROOT, f"proccessed_{animal_dir_name}_post")


def main():
    if not os.path.isdir(RAW_ROOT):
        raise RuntimeError(f"RAW_ROOT does not exist or is not a directory: {RAW_ROOT}")

    safe_mkdir(OUTPUT_ROOT)

    animal_dirs = sorted([
        d for d in os.listdir(RAW_ROOT)
        if os.path.isdir(os.path.join(RAW_ROOT, d))
    ])

    if not animal_dirs:
        print(f"No subfolders found in RAW_ROOT: {RAW_ROOT}")
        return

    # Build job list
    jobs = []
    for animal in animal_dirs:
        animal_path = os.path.join(RAW_ROOT, animal)
        vids = sorted(glob.glob(os.path.join(animal_path, VIDEO_GLOB)))
        if not vids:
            continue

        out_dir = animal_output_folder(animal)
        safe_mkdir(out_dir)

        for v in vids:
            jobs.append((animal, v, out_dir))

    if not jobs:
        print(f"No videos found matching '{VIDEO_GLOB}' under: {RAW_ROOT}")
        return

    print(f"Found {len(jobs)} video(s) across {len(set(a for a,_,_ in jobs))} animal folder(s).")
    print(f"RAW_ROOT:    {RAW_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"TRACKING:    {TRACKING_SCRIPT}")
    print("-" * 80)

    n_ok = 0
    n_fail = 0
    failures = []

    start_time = time.time()

    # tqdm progress bar (gives ETA automatically)
    with tqdm(total=len(jobs), desc="Tracking videos", unit="video") as pbar:
        for idx, (animal, video_path, out_dir) in enumerate(jobs, start=1):
            video_name = os.path.basename(video_path)

            # Show which file is currently running
            pbar.set_postfix({"animal": animal, "video": video_name})

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

            # Optional extra ETA estimate based on average so far (tqdm also has ETA)
            elapsed = time.time() - start_time
            avg_per_video = elapsed / idx
            remaining = avg_per_video * (len(jobs) - idx)
            pbar.set_postfix_str(f"{animal}/{video_name} | ETA ~ {remaining/60:.1f} min")

            pbar.update(1)

    print("\n" + "=" * 80)
    print("Pipeline complete")
    print(f"  Success: {n_ok}")
    print(f"  Failed:  {n_fail}")
    print(f"  Output:  {OUTPUT_ROOT}")

    if failures:
        # Write a simple failure log so you can rerun just the missing ones
        fail_log = os.path.join(OUTPUT_ROOT, "tracking_failures.txt")
        with open(fail_log, "w", encoding="utf-8") as f:
            for v in failures:
                f.write(v + "\n")
        print(f"\nWrote failure list to: {fail_log}")

    print("=" * 80)


if __name__ == "__main__":
    main()
