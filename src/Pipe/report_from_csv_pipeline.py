import os
import glob
import json
import base64
import subprocess
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===================== USER SETTINGS ===================== #
PROCESSED_ROOT = r"/Users/sowmithbonda/NeuroLab/Behaviour_Tracking_sb/data/proccessed"
CSV_GLOB = "**/*_ball_speed_fast.csv"     # recursive search inside PROCESSED_ROOT

# Reports will be saved as:
#   REPORTS_ROOT/<animal_folder>/<video>_report.html
#   REPORTS_ROOT/<animal_folder>/<video>_report.pdf
REPORTS_ROOT = r"/Users/sowmithbonda/NeuroLab/Behaviour_Tracking_sb/reports(0.06smoothed)"

SMOOTH_WINDOW_SEC = 0.06                  # smoothing window in seconds
MIN_FEATURES_GOOD = 10                   # quality threshold

# PDF settings (used by playwright if available)
PDF_FORMAT = "Letter"                    # "Letter" or "A4"
# ========================================================= #


def rolling_mean(x, w):
    if w <= 1:
        return x
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def fig_to_base64_png():
    """Convert the *current* matplotlib figure to a base64-encoded PNG string."""
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def make_plot_speed(t, speed, speed_s, smooth_label):
    plt.figure()
    plt.plot(t, speed, label="raw")
    plt.plot(t, speed_s, label=smooth_label)
    plt.title("Ball surface speed (cm/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    return fig_to_base64_png()


def make_plot_abs_omega(t, abs_omega, abs_omega_s, smooth_label):
    plt.figure()
    plt.plot(t, abs_omega, label="raw")
    plt.plot(t, abs_omega_s, label=smooth_label)
    plt.title("|ωz| (rad/s)")
    plt.xlabel("Time (s)")
    plt.legend()
    return fig_to_base64_png()


def make_plot_features(t, n_used, n_used_s, smooth_label):
    plt.figure()
    plt.plot(t, n_used, label="raw")
    plt.plot(t, n_used_s, label=smooth_label)
    plt.title("n_features_used (tracking quality)")
    plt.xlabel("Time (s)")
    plt.legend()
    return fig_to_base64_png()


def make_plot_omega_std(t, omega_std):
    plt.figure()
    plt.plot(t, omega_std)
    plt.title("ω std (rad/s) (noise / confidence proxy)")
    plt.xlabel("Time (s)")
    return fig_to_base64_png()


def make_hist_speed(speed):
    plt.figure()
    plt.hist(speed, bins=60)
    plt.title("Speed histogram (cm/s)")
    plt.xlabel("Speed (cm/s)")
    return fig_to_base64_png()


def compute_metrics(t, speed, abs_omega, omega_std, n_used):
    metrics = {
        "n_rows": int(len(speed)),
        "duration_sec": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "speed_mean": float(np.mean(speed)),
        "speed_median": float(np.median(speed)),
        "speed_p90": float(np.percentile(speed, 90)),
        "speed_p95": float(np.percentile(speed, 95)),
        "speed_max": float(np.max(speed)),
        "abs_omega_mean": float(np.mean(abs_omega)),
        "abs_omega_p95": float(np.percentile(abs_omega, 95)),
        "omega_std_mean": float(np.mean(omega_std)),
        "frac_low_features": float(np.mean(n_used < MIN_FEATURES_GOOD)),
        "frac_zero_speed": float(np.mean(speed == 0.0)),
        "distance_proxy_cm": float(np.trapz(speed, t)) if len(t) > 1 else 0.0,
    }
    return metrics


def video_name_from_csv(csv_path: str) -> str:
    """
    Convert:
      post_1_ball_speed_fast.csv -> post_1
      post_ball_speed_fast.csv   -> post
    """
    base = os.path.basename(csv_path)
    suffix = "_ball_speed_fast.csv"
    if base.endswith(suffix):
        return base[:-len(suffix)]
    return os.path.splitext(base)[0]


def animal_folder_from_csv(csv_path: str) -> str:
    """
    Your processed structure is:
      PROCESSED_ROOT/proccessed_het230_post/<csv>
    Report folder should be:
      REPORTS_ROOT/proccessed_het230_post/
    """
    return os.path.basename(os.path.dirname(csv_path))


def find_chrome_or_edge() -> str | None:
    """
    Locate Chrome or Edge on Windows. Returns full path or None.
    """
    candidates = [
        os.path.join(os.environ.get("PROGRAMFILES", ""), r"Google\Chrome\Application\chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), r"Google\Chrome\Application\chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES", ""), r"Microsoft\Edge\Application\msedge.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), r"Microsoft\Edge\Application\msedge.exe"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def html_to_pdf(html_path: str, pdf_path: str) -> bool:
    """
    Render local HTML file to PDF.
    Tries Playwright first. If unavailable, falls back to Chrome/Edge headless.
    Returns True on success.
    """
    # 1) Try Playwright (best reliability)
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto("file:///" + html_path.replace("\\", "/"))
            page.emulate_media(media="print")
            page.pdf(path=pdf_path, format=PDF_FORMAT, print_background=True)
            page.close()
            browser.close()
        return True
    except Exception:
        pass

    # 2) Fallback: Chrome/Edge headless
    chrome = find_chrome_or_edge()
    if not chrome:
        return False

    url = "file:///" + html_path.replace("\\", "/")
    cmd = [
        chrome,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def process_csv(csv_path: str) -> tuple[str, str]:
    df = pd.read_csv(csv_path)

    required = {"time_sec", "surface_speed_cm_per_s", "abs_omega_z_rad_per_s"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing required columns {missing}: {csv_path}")

    t = df["time_sec"].to_numpy(dtype=float)
    speed = df["surface_speed_cm_per_s"].to_numpy(dtype=float)
    abs_omega = df["abs_omega_z_rad_per_s"].to_numpy(dtype=float)

    omega_std = df["omega_std_rad_per_s"].to_numpy(dtype=float) if "omega_std_rad_per_s" in df.columns else np.zeros(len(df))
    n_used = df["n_features_used"].to_numpy(dtype=float) if "n_features_used" in df.columns else np.zeros(len(df))

    # Estimate sampling rate from time steps
    dt = np.diff(t)
    dt_med = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.0
    fs = (1.0 / dt_med) if dt_med > 0 else 30.0

    w = max(1, int(round(SMOOTH_WINDOW_SEC * fs)))
    smooth_label = f"smoothed ({SMOOTH_WINDOW_SEC:.2f}s)"

    speed_s = rolling_mean(speed, w)
    abs_omega_s = rolling_mean(abs_omega, w)
    n_used_s = rolling_mean(n_used, w)

    metrics = compute_metrics(t, speed, abs_omega, omega_std, n_used)
    metrics["fs_est_hz"] = float(fs)
    metrics["smooth_window_samples"] = int(w)

    # Embedded plots
    img_speed = make_plot_speed(t, speed, speed_s, smooth_label)
    img_omega = make_plot_abs_omega(t, abs_omega, abs_omega_s, smooth_label)
    img_feat = make_plot_features(t, n_used, n_used_s, smooth_label)
    img_std = make_plot_omega_std(t, omega_std)
    img_hist = make_hist_speed(speed)

    # Output paths: flat per animal folder, named after video
    animal_folder = animal_folder_from_csv(csv_path)
    video_name = video_name_from_csv(csv_path)

    out_dir = os.path.join(REPORTS_ROOT, animal_folder)
    os.makedirs(out_dir, exist_ok=True)

    html_path = os.path.join(out_dir, f"{video_name}_report.html")
    pdf_path = os.path.join(out_dir, f"{video_name}_report.pdf")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Ball Tracking Report - {video_name}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    img {{ width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
    pre {{ background: #f6f6f6; padding: 12px; border-radius: 8px; overflow-x: auto; }}
    code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Ball Tracking Report</h1>
  <p><b>Animal folder:</b> <code>{animal_folder}</code></p>
  <p><b>Video:</b> <code>{video_name}</code></p>
  <p><b>CSV:</b> <code>{csv_path}</code></p>

  <h2>Summary metrics</h2>
  <pre>{json.dumps(metrics, indent=2)}</pre>

  <h2>Plots</h2>
  <div class="grid">
    <div>
      <h3>Speed vs time</h3>
      <img src="data:image/png;base64,{img_speed}">
    </div>
    <div>
      <h3>|ω| vs time</h3>
      <img src="data:image/png;base64,{img_omega}">
    </div>
    <div>
      <h3>Features used vs time</h3>
      <img src="data:image/png;base64,{img_feat}">
    </div>
    <div>
      <h3>ω std vs time</h3>
      <img src="data:image/png;base64,{img_std}">
    </div>
  </div>

  <h2>Distribution</h2>
  <div style="max-width: 800px;">
    <h3>Speed histogram</h3>
    <img src="data:image/png;base64,{img_hist}">
  </div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    ok_pdf = html_to_pdf(html_path, pdf_path)
    if not ok_pdf:
        # Still return html path; pdf path exists only if conversion worked
        raise RuntimeError(
            "Generated HTML but could not generate PDF. "
            "Install Playwright (recommended): pip install playwright && python -m playwright install chromium "
            "OR ensure Chrome/Edge is installed."
        )

    return html_path, pdf_path


def main():
    os.makedirs(REPORTS_ROOT, exist_ok=True)

    pattern = os.path.join(PROCESSED_ROOT, CSV_GLOB)
    csvs = sorted(glob.glob(pattern, recursive=True))
    if not csvs:
        print(f"No CSVs found under: {PROCESSED_ROOT}")
        print(f"Pattern: {CSV_GLOB}")
        return

    print(f"Found {len(csvs)} CSV(s). Writing HTML+PDF reports under: {REPORTS_ROOT}")
    ok = 0
    fail = 0

    for csv_path in csvs:
        try:
            html_path, pdf_path = process_csv(csv_path)
            print(f"✅ {os.path.basename(csv_path)} -> {os.path.basename(html_path)} + {os.path.basename(pdf_path)}")
            ok += 1
        except Exception as e:
            print(f"❌ Failed on {csv_path}\n   {e}")
            fail += 1

    print("\nDone.")
    print(f"Success: {ok}")
    print(f"Failed:  {fail}")


if __name__ == "__main__":
    main()
