import os
import glob
import math
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ===================== USER SETTINGS ===================== #
PROCESSED_ROOT = r"/Users/sowmithbonda/NeuroLab/Behaviour_Tracking_sb/data/proccessed"
CSV_GLOB = "**/*_ball_speed_fast.csv"     # recursive

REPORTS_ROOT = r"/Users/sowmithbonda/NeuroLab/Behaviour_Tracking_sb/data/reports"

THRESH_CM_S = 1.0                        # threshold for bouts/timing
SMOOTH_WINDOW_SEC = 0.5                  # smoothing for plots only (0 disables)

# PDF layout
PAGE_SIZE = letter
MARGIN_IN = 0.6
PLOT_HEIGHT_IN = 2.6                     # each plot height (full width)
DPI = 200                                # plot render dpi

# Cap number of rows written to bout CSVs (safety)
MAX_BOUT_ROWS = 2_000_000
# ========================================================= #


# ---------- naming helpers ----------

def video_name_from_csv(csv_path: str) -> str:
    """
    post_1_ball_speed_fast.csv -> post_1
    post_ball_speed_fast.csv   -> post
    """
    base = os.path.basename(csv_path)
    suffix = "_ball_speed_fast.csv"
    if base.endswith(suffix):
        return base[:-len(suffix)]
    return os.path.splitext(base)[0]


def animal_folder_from_csv(csv_path: str) -> str:
    # CSV lives in: .../proccessed_het230_post/<csv>
    return os.path.basename(os.path.dirname(csv_path))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------- threshold / bout logic ----------

def compute_durations_by_state(t: np.ndarray, state: np.ndarray) -> Tuple[float, float]:
    """
    Integrate time spent above/below using dt between samples.
    Assign dt[i] = t[i+1]-t[i] to state[i] (left-hand rule).
    """
    if len(t) < 2:
        return 0.0, 0.0
    dt = np.diff(t)
    dt = np.where(dt > 0, dt, 0.0)

    s_left = state[:-1].astype(bool)
    above = float(np.sum(dt[s_left]))
    below = float(np.sum(dt[~s_left]))
    return above, below


def bouts_from_state(t: np.ndarray, state: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    Return alternating contiguous bouts across entire recording:
      [(start, end, duration), ...]
    Transition time is taken as the timestamp of the first sample in the new state.
    """
    n = len(t)
    if n == 0:
        return []

    bouts = []
    in_state = bool(state[0])
    start_t = float(t[0])

    for i in range(1, n):
        s = bool(state[i])
        if s != in_state:
            end_t = float(t[i])
            bouts.append((start_t, end_t, max(0.0, end_t - start_t)))
            start_t = float(t[i])
            in_state = s

    end_t = float(t[-1])
    bouts.append((start_t, end_t, max(0.0, end_t - start_t)))
    return bouts


def split_bouts(all_bouts: List[Tuple[float, float, float]], initial_state: bool) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    all_bouts alternates starting with initial_state.
    Return (true_bouts, false_bouts).
    """
    true_bouts = []
    false_bouts = []
    st = initial_state
    for b in all_bouts:
        (true_bouts if st else false_bouts).append(b)
        st = not st
    return true_bouts, false_bouts


def write_bouts_csv(path: str, bouts: List[Tuple[float, float, float]]) -> None:
    if len(bouts) > MAX_BOUT_ROWS:
        bouts = bouts[:MAX_BOUT_ROWS]
    df = pd.DataFrame(bouts, columns=["start_sec", "end_sec", "duration_sec"])
    df.to_csv(path, index=False)


# ---------- plotting ----------

def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def make_plot_image_reader(
    t: np.ndarray,
    y: np.ndarray,
    title: str,
    y_label: str,
    threshold: Optional[float] = None,
    y_smooth: Optional[np.ndarray] = None,
    smooth_label: str = "smoothed",
) -> ImageReader:
    """
    Renders a single matplotlib plot to an in-memory PNG and returns ReportLab ImageReader.
    """
    # Wide plot so it fills PDF width cleanly
    plt.figure(figsize=(11, PLOT_HEIGHT_IN), dpi=DPI)
    plt.plot(t, y, label="raw", linewidth=1)

    if y_smooth is not None:
        plt.plot(t, y_smooth, label=smooth_label, linewidth=2)
        plt.legend(loc="upper right")

    if threshold is not None:
        plt.axhline(threshold, linestyle="--", linewidth=1)
        # keep the threshold line unlabeled to reduce clutter

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(y_label)

    # Save to bytes
    import io
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return ImageReader(buf)


# ---------- metrics ----------

def compute_summary_metrics(
    t: np.ndarray,
    speed: np.ndarray,
    ang: np.ndarray,
    total_above: float,
    total_below: float,
    above_bouts: List[Tuple[float, float, float]],
    n_used: Optional[np.ndarray],
    omega_std: Optional[np.ndarray],
) -> Dict[str, float]:
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    metrics = {
        "duration_sec": duration,
        "time_above_threshold_sec": float(total_above),
        "time_below_threshold_sec": float(total_below),
        "pct_time_above_threshold": float((total_above / duration) * 100.0) if duration > 0 else 0.0,

        "speed_mean": float(np.mean(speed)),
        "speed_median": float(np.median(speed)),
        "speed_p95": float(np.percentile(speed, 95)),
        "speed_max": float(np.max(speed)),

        "angular_speed_mean_rad_s": float(np.mean(ang)),
        "angular_speed_p95_rad_s": float(np.percentile(ang, 95)),
        "angular_speed_max_rad_s": float(np.max(ang)),

        "distance_proxy_cm": float(np.trapz(speed, t)) if len(t) > 1 else 0.0,

        "n_above_bouts": float(len(above_bouts)),
        "above_bout_mean_sec": float(np.mean([b[2] for b in above_bouts])) if above_bouts else 0.0,
        "above_bout_max_sec": float(np.max([b[2] for b in above_bouts])) if above_bouts else 0.0,
    }

    if n_used is not None and len(n_used) == len(t):
        metrics["n_features_used_mean"] = float(np.mean(n_used))
        metrics["n_features_used_p10"] = float(np.percentile(n_used, 10))
        metrics["n_features_used_min"] = float(np.min(n_used))

    if omega_std is not None and len(omega_std) == len(t):
        metrics["omega_std_mean_rad_s"] = float(np.mean(omega_std))
        metrics["omega_std_p95_rad_s"] = float(np.percentile(omega_std, 95))

    return metrics


# ---------- PDF generation ----------

def make_pdf_report(
    pdf_path: str,
    animal_folder: str,
    video_name: str,
    csv_path: str,
    metrics: Dict[str, float],
    speed_plot: ImageReader,
    ang_plot: ImageReader,
) -> None:
    c = canvas.Canvas(pdf_path, pagesize=PAGE_SIZE)
    page_w, page_h = PAGE_SIZE

    margin = MARGIN_IN * inch
    usable_w = page_w - 2 * margin
    y = page_h - margin

    # Title block
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Ball Speed Report")
    y -= 0.25 * inch

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Animal folder: {animal_folder}")
    y -= 0.18 * inch
    c.drawString(margin, y, f"Video: {video_name}")
    y -= 0.18 * inch
    c.drawString(margin, y, f"CSV: {csv_path}")
    y -= 0.22 * inch

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Threshold: {THRESH_CM_S:.3f} cm/s")
    y -= 0.25 * inch

    # Key threshold totals
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Threshold timing")
    y -= 0.18 * inch

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Time >= {THRESH_CM_S:.3f} cm/s: {metrics['time_above_threshold_sec']:.3f} s")
    y -= 0.16 * inch
    c.drawString(margin, y, f"Time <  {THRESH_CM_S:.3f} cm/s: {metrics['time_below_threshold_sec']:.3f} s")
    y -= 0.16 * inch
    c.drawString(margin, y, f"Percent time above: {metrics['pct_time_above_threshold']:.2f}%")
    y -= 0.24 * inch

    # Other metrics (compact)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Other metrics")
    y -= 0.18 * inch

    c.setFont("Helvetica", 10)
    lines = [
        f"Speed (cm/s): mean={metrics['speed_mean']:.3f}, median={metrics['speed_median']:.3f}, p95={metrics['speed_p95']:.3f}, max={metrics['speed_max']:.3f}",
        f"Angular speed (rad/s): mean={metrics['angular_speed_mean_rad_s']:.3f}, p95={metrics['angular_speed_p95_rad_s']:.3f}, max={metrics['angular_speed_max_rad_s']:.3f}",
        f"Distance proxy (∫speed dt): {metrics['distance_proxy_cm']:.3f} cm",
        f"Above-threshold bouts: n={int(metrics['n_above_bouts'])}, mean={metrics['above_bout_mean_sec']:.3f}s, max={metrics['above_bout_max_sec']:.3f}s",
    ]

    # Optional quality lines
    if "n_features_used_mean" in metrics:
        lines.append(
            f"Features used: mean={metrics['n_features_used_mean']:.2f}, p10={metrics['n_features_used_p10']:.2f}, min={metrics['n_features_used_min']:.2f}"
        )
    if "omega_std_mean_rad_s" in metrics:
        lines.append(
            f"Omega std (rad/s): mean={metrics['omega_std_mean_rad_s']:.3f}, p95={metrics['omega_std_p95_rad_s']:.3f}"
        )

    for ln in lines:
        c.drawString(margin, y, ln[:160])  # prevent super long lines
        y -= 0.16 * inch

    y -= 0.20 * inch

    # Plots: full width, stacked
    plot_h = PLOT_HEIGHT_IN * inch

    # If not enough space, new page
    if y - plot_h < margin:
        c.showPage()
        y = page_h - margin

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "1) Surface speed vs time")
    y -= 0.18 * inch
    c.drawImage(speed_plot, margin, y - plot_h, width=usable_w, height=plot_h, preserveAspectRatio=True, mask='auto')
    y -= plot_h + 0.35 * inch

    if y - plot_h < margin:
        c.showPage()
        y = page_h - margin

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "2) Angular speed vs time")
    y -= 0.18 * inch
    c.drawImage(ang_plot, margin, y - plot_h, width=usable_w, height=plot_h, preserveAspectRatio=True, mask='auto')
    y -= plot_h + 0.25 * inch

    c.save()


# ---------- core per-csv ----------

def process_csv(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    if "time_sec" not in df.columns or "surface_speed_cm_per_s" not in df.columns:
        raise RuntimeError(f"CSV missing required columns: {csv_path}")

    t_all = df["time_sec"].to_numpy(dtype=float)
    speed_all = df["surface_speed_cm_per_s"].to_numpy(dtype=float)

    # Angular speed: prefer abs_omega, else abs(omega)
    if "abs_omega_z_rad_per_s" in df.columns:
        ang_all = df["abs_omega_z_rad_per_s"].to_numpy(dtype=float)
    elif "omega_z_rad_per_s" in df.columns:
        ang_all = np.abs(df["omega_z_rad_per_s"].to_numpy(dtype=float))
    else:
        ang_all = np.zeros_like(speed_all)

    n_used_all = df["n_features_used"].to_numpy(dtype=float) if "n_features_used" in df.columns else None
    omega_std_all = df["omega_std_rad_per_s"].to_numpy(dtype=float) if "omega_std_rad_per_s" in df.columns else None

    # Filter finite
    finite = np.isfinite(t_all) & np.isfinite(speed_all) & np.isfinite(ang_all)
    t = t_all[finite]
    speed = speed_all[finite]
    ang = ang_all[finite]
    n_used = n_used_all[finite] if n_used_all is not None else None
    omega_std = omega_std_all[finite] if omega_std_all is not None else None

    if len(t) == 0:
        raise RuntimeError(f"No valid rows after filtering NaNs: {csv_path}")

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    speed = speed[order]
    ang = ang[order]
    if n_used is not None:
        n_used = n_used[order]
    if omega_std is not None:
        omega_std = omega_std[order]

    # Smooth for plots only
    if SMOOTH_WINDOW_SEC > 0 and len(t) > 1:
        dt = np.diff(t)
        dt_med = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.0
        fs = (1.0 / dt_med) if dt_med > 0 else 30.0
        w = max(1, int(round(SMOOTH_WINDOW_SEC * fs)))
        speed_s = rolling_mean(speed, w)
        ang_s = rolling_mean(ang, w)
        smooth_label = f"smoothed ({SMOOTH_WINDOW_SEC:.2f}s)"
    else:
        speed_s = None
        ang_s = None
        smooth_label = "smoothed"

    # Threshold bouts based on RAW speed
    state_above = speed >= THRESH_CM_S
    total_above, total_below = compute_durations_by_state(t, state_above)

    all_bouts = bouts_from_state(t, state_above)
    initial_state = bool(state_above[0])
    above_bouts, below_bouts = split_bouts(all_bouts, initial_state=initial_state)

    # Output folder: REPORTS_ROOT/<animal>/<video>/
    animal_folder = animal_folder_from_csv(csv_path)
    video_name = video_name_from_csv(csv_path)
    out_dir = os.path.join(REPORTS_ROOT, animal_folder, video_name)
    ensure_dir(out_dir)

    # Write bout CSVs
    above_csv = os.path.join(out_dir, "above_threshold_bouts.csv")
    below_csv = os.path.join(out_dir, "below_threshold_bouts.csv")
    write_bouts_csv(above_csv, above_bouts)
    write_bouts_csv(below_csv, below_bouts)

    # Plots for PDF (full width)
    speed_plot = make_plot_image_reader(
        t=t,
        y=speed,
        y_smooth=speed_s,
        smooth_label=smooth_label,
        title="Surface speed vs time",
        y_label="Speed (cm/s)",
        threshold=THRESH_CM_S
    )
    ang_plot = make_plot_image_reader(
        t=t,
        y=ang,
        y_smooth=ang_s,
        smooth_label=smooth_label,
        title="Angular speed vs time",
        y_label="Angular speed (rad/s)",
        threshold=None
    )

    metrics = compute_summary_metrics(
        t=t,
        speed=speed,
        ang=ang,
        total_above=total_above,
        total_below=total_below,
        above_bouts=above_bouts,
        n_used=n_used,
        omega_std=omega_std
    )

    pdf_path = os.path.join(out_dir, "report.pdf")
    make_pdf_report(
        pdf_path=pdf_path,
        animal_folder=animal_folder,
        video_name=video_name,
        csv_path=csv_path,
        metrics=metrics,
        speed_plot=speed_plot,
        ang_plot=ang_plot
    )


def main():
    ensure_dir(REPORTS_ROOT)
    pattern = os.path.join(PROCESSED_ROOT, CSV_GLOB)
    csvs = sorted(glob.glob(pattern, recursive=True))

    if not csvs:
        print(f"No CSVs found under: {PROCESSED_ROOT}")
        print(f"Pattern: {CSV_GLOB}")
        return

    print(f"Found {len(csvs)} CSV(s). Writing reports into: {REPORTS_ROOT}")
    ok = 0
    fail = 0

    for csv_path in csvs:
        try:
            process_csv(csv_path)
            print(f"✅ {os.path.basename(csv_path)} -> report.pdf + bout CSVs")
            ok += 1
        except Exception as e:
            print(f"❌ Failed on {csv_path}\n   {e}")
            fail += 1

    print("\nDone.")
    print(f"Success: {ok}")
    print(f"Failed:  {fail}")


if __name__ == "__main__":
    main()
