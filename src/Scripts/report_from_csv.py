import os
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ---------------- Defaults (can be overridden by CLI) ---------------- #
DEFAULT_THRESH_CM_S = 1.0
DEFAULT_SMOOTH_WINDOW_SEC = 0.5

PAGE_SIZE = letter
MARGIN_IN = 0.6
PLOT_HEIGHT_IN = 2.6
DPI = 200
MAX_BOUT_ROWS = 2_000_000
# --------------------------------------------------------------------- #


def video_name_from_csv(csv_path: str) -> str:
    """
    induction_ball_speed_fast.csv -> induction
    post 3_ball_speed_fast.csv    -> post 3
    """
    base = os.path.basename(csv_path)
    suffix = "_ball_speed_fast.csv"
    if base.endswith(suffix):
        return base[:-len(suffix)]
    return os.path.splitext(base)[0]


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


def split_bouts(
    all_bouts: List[Tuple[float, float, float]],
    initial_state: bool
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    all_bouts alternates starting with initial_state.
    Return (true_bouts, false_bouts).
    """
    true_bouts: List[Tuple[float, float, float]] = []
    false_bouts: List[Tuple[float, float, float]] = []
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
) -> ImageReader:
    """
    Renders a single matplotlib plot to an in-memory PNG and returns ReportLab ImageReader.
    """
    plt.figure(figsize=(11, PLOT_HEIGHT_IN), dpi=DPI)
    plt.plot(t, y, linewidth=1.5)

    if threshold is not None:
        plt.axhline(threshold, linestyle="--", linewidth=1)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(y_label)

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
    speed_for_metrics: np.ndarray,
    ang_for_metrics: np.ndarray,
    total_above: float,
    total_below: float,
    above_bouts: List[Tuple[float, float, float]],
    n_used: Optional[np.ndarray],
    omega_std: Optional[np.ndarray],
) -> Dict[str, float]:
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    metrics: Dict[str, float] = {
        "duration_sec": duration,
        "time_above_threshold_sec": float(total_above),
        "time_below_threshold_sec": float(total_below),
        "pct_time_above_threshold": float((total_above / duration) * 100.0) if duration > 0 else 0.0,

        "speed_mean": float(np.mean(speed_for_metrics)),
        "speed_median": float(np.median(speed_for_metrics)),
        "speed_p95": float(np.percentile(speed_for_metrics, 95)),
        "speed_max": float(np.max(speed_for_metrics)),

        "angular_speed_mean_rad_s": float(np.mean(ang_for_metrics)),
        "angular_speed_median_rad_s": float(np.median(ang_for_metrics)),
        "angular_speed_p95_rad_s": float(np.percentile(ang_for_metrics, 95)),
        "angular_speed_max_rad_s": float(np.max(ang_for_metrics)),

        "distance_proxy_cm": float(np.trapz(speed_for_metrics, t)) if len(t) > 1 else 0.0,

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

def _add_plot_page_if_needed(c: canvas.Canvas, y: float, plot_h: float, margin: float, page_h: float) -> float:
    if y - plot_h < margin:
        c.showPage()
        return page_h - margin
    return y


def make_pdf_report(
    pdf_path: str,
    rel_parent_dir: str,
    video_name: str,
    csv_path: str,
    thresh_cm_s: float,
    metrics: Dict[str, float],
    speed_raw_plot: ImageReader,
    speed_smooth_plot: ImageReader,
    ang_raw_plot: ImageReader,
    ang_smooth_plot: ImageReader,
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
    c.drawString(margin, y, f"Processed subpath: {rel_parent_dir}")
    y -= 0.18 * inch
    c.drawString(margin, y, f"Video: {video_name}")
    y -= 0.18 * inch
    c.drawString(margin, y, f"CSV: {csv_path}")
    y -= 0.22 * inch

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Threshold (applied to SMOOTHED speed): {thresh_cm_s:.3f} cm/s")
    y -= 0.25 * inch

    # Threshold totals
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Threshold timing (from smoothed speed)")
    y -= 0.18 * inch

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Time >= {thresh_cm_s:.3f} cm/s: {metrics['time_above_threshold_sec']:.3f} s")
    y -= 0.16 * inch
    c.drawString(margin, y, f"Time <  {thresh_cm_s:.3f} cm/s: {metrics['time_below_threshold_sec']:.3f} s")
    y -= 0.16 * inch
    c.drawString(margin, y, f"Percent time above: {metrics['pct_time_above_threshold']:.2f}%")
    y -= 0.24 * inch

    # Other metrics
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, y, "Other metrics (computed on smoothed speed)")
    y -= 0.18 * inch

    c.setFont("Helvetica", 10)
    lines = [
        f"Speed (cm/s): mean={metrics['speed_mean']:.3f}, median={metrics['speed_median']:.3f}, p95={metrics['speed_p95']:.3f}, max={metrics['speed_max']:.3f}",
        f"Angular speed (rad/s): mean={metrics['angular_speed_mean_rad_s']:.3f}, median={metrics['angular_speed_median_rad_s']:.3f}, p95={metrics['angular_speed_p95_rad_s']:.3f}, max={metrics['angular_speed_max_rad_s']:.3f}",
        f"Distance proxy (∫speed dt): {metrics['distance_proxy_cm']:.3f} cm",
        f"Above-threshold bouts: n={int(metrics['n_above_bouts'])}, mean={metrics['above_bout_mean_sec']:.3f}s, max={metrics['above_bout_max_sec']:.3f}s",
    ]
    if "n_features_used_mean" in metrics:
        lines.append(
            f"Features used: mean={metrics['n_features_used_mean']:.2f}, p10={metrics['n_features_used_p10']:.2f}, min={metrics['n_features_used_min']:.2f}"
        )
    if "omega_std_mean_rad_s" in metrics:
        lines.append(
            f"Omega std (rad/s): mean={metrics['omega_std_mean_rad_s']:.3f}, p95={metrics['omega_std_p95_rad_s']:.3f}"
        )

    for ln in lines:
        c.drawString(margin, y, ln[:160])
        y -= 0.16 * inch

    y -= 0.20 * inch

    plot_h = PLOT_HEIGHT_IN * inch

    # 1) Surface speed raw
    y = _add_plot_page_if_needed(c, y, plot_h, margin, page_h)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "1) Surface speed (RAW) vs time")
    y -= 0.18 * inch
    c.drawImage(speed_raw_plot, margin, y - plot_h, width=usable_w, height=plot_h, preserveAspectRatio=True, mask="auto")
    y -= plot_h + 0.35 * inch

    # 2) Surface speed smoothed
    y = _add_plot_page_if_needed(c, y, plot_h, margin, page_h)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "2) Surface speed (SMOOTHED) vs time")
    y -= 0.18 * inch
    c.drawImage(speed_smooth_plot, margin, y - plot_h, width=usable_w, height=plot_h, preserveAspectRatio=True, mask="auto")
    y -= plot_h + 0.35 * inch

    # 3) Angular speed raw
    y = _add_plot_page_if_needed(c, y, plot_h, margin, page_h)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "3) Angular speed (RAW) vs time")
    y -= 0.18 * inch
    c.drawImage(ang_raw_plot, margin, y - plot_h, width=usable_w, height=plot_h, preserveAspectRatio=True, mask="auto")
    y -= plot_h + 0.35 * inch

    # 4) Angular speed smoothed
    y = _add_plot_page_if_needed(c, y, plot_h, margin, page_h)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "4) Angular speed (SMOOTHED) vs time")
    y -= 0.18 * inch
    c.drawImage(ang_smooth_plot, margin, y - plot_h, width=usable_w, height=plot_h, preserveAspectRatio=True, mask="auto")
    y -= plot_h + 0.25 * inch

    c.save()


def process_one_csv(
    csv_path: str,
    processed_root: str,
    reports_root: str,
    thresh_cm_s: float,
    smooth_window_sec: float,
) -> str:
    """
    Processes one *_ball_speed_fast.csv and writes:
      report.pdf
      above_threshold_bouts.csv
      below_threshold_bouts.csv

    Changes from previous version:
      - PDF includes separate raw + smoothed plots for both signals
      - Threshold timing/bouts are computed on SMOOTHED surface speed
    """
    df = pd.read_csv(csv_path)

    if "time_sec" not in df.columns or "surface_speed_cm_per_s" not in df.columns:
        raise RuntimeError(f"CSV missing required columns: {csv_path}")

    t_all = df["time_sec"].to_numpy(dtype=float)
    speed_all = df["surface_speed_cm_per_s"].to_numpy(dtype=float)

    # Angular speed preference
    if "abs_omega_z_rad_per_s" in df.columns:
        ang_all = df["abs_omega_z_rad_per_s"].to_numpy(dtype=float)
    elif "omega_z_rad_per_s" in df.columns:
        ang_all = np.abs(df["omega_z_rad_per_s"].to_numpy(dtype=float))
    else:
        ang_all = np.zeros_like(speed_all)

    n_used_all = df["n_features_used"].to_numpy(dtype=float) if "n_features_used" in df.columns else None
    omega_std_all = df["omega_std_rad_per_s"].to_numpy(dtype=float) if "omega_std_rad_per_s" in df.columns else None

    finite = np.isfinite(t_all) & np.isfinite(speed_all) & np.isfinite(ang_all)
    t = t_all[finite]
    speed_raw = speed_all[finite]
    ang_raw = ang_all[finite]
    n_used = n_used_all[finite] if n_used_all is not None else None
    omega_std = omega_std_all[finite] if omega_std_all is not None else None

    if len(t) == 0:
        raise RuntimeError(f"No valid rows after filtering NaNs: {csv_path}")

    order = np.argsort(t)
    t = t[order]
    speed_raw = speed_raw[order]
    ang_raw = ang_raw[order]
    if n_used is not None:
        n_used = n_used[order]
    if omega_std is not None:
        omega_std = omega_std[order]

    # Build smoothed signals (used for threshold + also plotted)
    if smooth_window_sec > 0 and len(t) > 1:
        dt = np.diff(t)
        dt_med = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.0
        fs = (1.0 / dt_med) if dt_med > 0 else 30.0
        w = max(1, int(round(smooth_window_sec * fs)))
        speed_smooth = rolling_mean(speed_raw, w)
        ang_smooth = rolling_mean(ang_raw, w)
    else:
        # If smoothing disabled, "smoothed" == raw
        speed_smooth = speed_raw.copy()
        ang_smooth = ang_raw.copy()

    # -------- CHANGE: threshold logic uses SMOOTHED surface speed --------
    state_above = speed_smooth >= thresh_cm_s
    total_above, total_below = compute_durations_by_state(t, state_above)

    all_bouts = bouts_from_state(t, state_above)
    initial_state = bool(state_above[0])
    above_bouts, below_bouts = split_bouts(all_bouts, initial_state=initial_state)

    # Mirror processed folder structure
    rel_csv = os.path.relpath(csv_path, processed_root)
    rel_dir = os.path.dirname(rel_csv)  # cohort/date/animal

    video_name = video_name_from_csv(csv_path)
    out_dir = os.path.join(reports_root, rel_dir, video_name)
    ensure_dir(out_dir)

    # Write timestamp CSVs (from SMOOTHED thresholding)
    above_csv = os.path.join(out_dir, "above_threshold_bouts.csv")
    below_csv = os.path.join(out_dir, "below_threshold_bouts.csv")
    write_bouts_csv(above_csv, above_bouts)
    write_bouts_csv(below_csv, below_bouts)

    # Plots (separate raw and smoothed)
    speed_raw_plot = make_plot_image_reader(
        t=t,
        y=speed_raw,
        title="Surface speed (RAW) vs time",
        y_label="Speed (cm/s)",
        threshold=thresh_cm_s,
    )
    speed_smooth_plot = make_plot_image_reader(
        t=t,
        y=speed_smooth,
        title=f"Surface speed (SMOOTHED, {smooth_window_sec:.2f}s) vs time" if smooth_window_sec > 0 else "Surface speed (SMOOTHED=RAW) vs time",
        y_label="Speed (cm/s)",
        threshold=thresh_cm_s,
    )
    ang_raw_plot = make_plot_image_reader(
        t=t,
        y=ang_raw,
        title="Angular speed (RAW) vs time",
        y_label="Angular speed (rad/s)",
        threshold=None,
    )
    ang_smooth_plot = make_plot_image_reader(
        t=t,
        y=ang_smooth,
        title=f"Angular speed (SMOOTHED, {smooth_window_sec:.2f}s) vs time" if smooth_window_sec > 0 else "Angular speed (SMOOTHED=RAW) vs time",
        y_label="Angular speed (rad/s)",
        threshold=None,
    )

    # -------- Metrics: compute on SMOOTHED speed (since thresholding is smoothed) --------
    metrics = compute_summary_metrics(
        t=t,
        speed_for_metrics=speed_smooth,
        ang_for_metrics=ang_smooth,  # optional; keeps consistent "smoothed summary"
        total_above=total_above,
        total_below=total_below,
        above_bouts=above_bouts,
        n_used=n_used,
        omega_std=omega_std,
    )

    pdf_path = os.path.join(out_dir, "report.pdf")
    make_pdf_report(
        pdf_path=pdf_path,
        rel_parent_dir=rel_dir,
        video_name=video_name,
        csv_path=csv_path,
        thresh_cm_s=thresh_cm_s,
        metrics=metrics,
        speed_raw_plot=speed_raw_plot,
        speed_smooth_plot=speed_smooth_plot,
        ang_raw_plot=ang_raw_plot,
        ang_smooth_plot=ang_smooth_plot,
    )

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Generate PDF + bout CSVs from one *_ball_speed_fast.csv")
    parser.add_argument("--csv", required=True, help="Path to one *_ball_speed_fast.csv")
    parser.add_argument("--processed-root", required=True, help="Root folder containing processed CSVs (used for mirroring structure)")
    parser.add_argument("--reports-root", required=True, help="Root folder to write reports into")
    parser.add_argument("--thresh", type=float, default=DEFAULT_THRESH_CM_S, help="Speed threshold in cm/s (default: 1.0)")
    parser.add_argument("--smooth-sec", type=float, default=DEFAULT_SMOOTH_WINDOW_SEC, help="Smoothing window for plots + thresholding (sec)")

    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    processed_root = os.path.abspath(args.processed_root)
    reports_root = os.path.abspath(args.reports_root)

    ensure_dir(reports_root)

    out_dir = process_one_csv(
        csv_path=csv_path,
        processed_root=processed_root,
        reports_root=reports_root,
        thresh_cm_s=float(args.thresh),
        smooth_window_sec=float(args.smooth_sec),
    )

    print(f"✅ Wrote report package: {out_dir}")


if __name__ == "__main__":
    main()
