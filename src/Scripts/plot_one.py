import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


THRESH_CM_S = 1.0       # draw threshold line
SMOOTH_WINDOW_SEC = 0.5  # 0 to disable smoothing


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def main():
    parser = argparse.ArgumentParser(description="Plot ball speed + angular speed from one CSV")
    parser.add_argument("--csv", required=True, help="Path to *_ball_speed_fast.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if "time_sec" not in df or "surface_speed_cm_per_s" not in df:
        raise RuntimeError("CSV must contain time_sec and surface_speed_cm_per_s")

    t = df["time_sec"].to_numpy(dtype=float)
    speed = df["surface_speed_cm_per_s"].to_numpy(dtype=float)

    # Angular speed logic (same as report pipeline)
    if "abs_omega_z_rad_per_s" in df:
        ang = df["abs_omega_z_rad_per_s"].to_numpy(dtype=float)
    elif "omega_z_rad_per_s" in df:
        ang = np.abs(df["omega_z_rad_per_s"].to_numpy(dtype=float))
    else:
        ang = np.zeros_like(speed)

    # Remove NaNs
    finite = np.isfinite(t) & np.isfinite(speed) & np.isfinite(ang)
    t = t[finite]
    speed = speed[finite]
    ang = ang[finite]

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    speed = speed[order]
    ang = ang[order]

    # Optional smoothing (for display only)
    if SMOOTH_WINDOW_SEC > 0 and len(t) > 1:
        dt = np.diff(t)
        dt_med = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.0
        fs = (1.0 / dt_med) if dt_med > 0 else 30.0
        w = max(1, int(round(SMOOTH_WINDOW_SEC * fs)))
        speed_s = rolling_mean(speed, w)
        ang_s = rolling_mean(ang, w)
    else:
        speed_s = None
        ang_s = None

    # -------- Plot 1: Surface speed --------
    plt.figure(figsize=(10, 4))
    plt.plot(t, speed, label="raw", linewidth=1)
    if speed_s is not None:
        plt.plot(t, speed_s, label=f"smoothed ({SMOOTH_WINDOW_SEC}s)", linewidth=2)
    plt.axhline(THRESH_CM_S, linestyle="--", linewidth=1, label="threshold")
    plt.xlabel("Time (s)")
    plt.ylabel("Surface speed (cm/s)")
    plt.title("Surface speed vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------- Plot 2: Angular speed --------
    plt.figure(figsize=(10, 4))
    plt.plot(t, ang, label="raw", linewidth=1)
    if ang_s is not None:
        plt.plot(t, ang_s, label=f"smoothed ({SMOOTH_WINDOW_SEC}s)", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Angular speed (rad/s)")
    plt.title("Angular speed vs time")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
