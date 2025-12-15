import cv2
import numpy as np
import math
import csv
import os
import json
import argparse


# ===================== CLI / USER SETTINGS ===================== #
def parse_args():
    parser = argparse.ArgumentParser(description="Ball rotation tracking (batch/pipeline friendly).")
    parser.add_argument("--video", required=True, help="Path to input video (post*.mp4).")
    parser.add_argument("--outdir", default=None, help="Directory to write outputs (CSV). If omitted, writes next to video.")
    parser.add_argument("--config", default=None, help="Path to ball config JSON. If omitted, uses DEFAULT_CONFIG_PATH.")
    return parser.parse_args()


args = parse_args()
VIDEO_PATH = args.video
OUTDIR = args.outdir

# Physical radius of ball (cm)
BALL_RADIUS_CM = 3.2

# Default config path if --config not provided
DEFAULT_CONFIG_PATH = r"C:\Users\16083\Coding\Behaviour_Tracking_Self\data\proccessed\ball_config.json"
BALL_CONFIG_PATH = args.config if args.config else DEFAULT_CONFIG_PATH

# Debug / FPS
SHOW_DEBUG = False              # keep False for speed (batch mode)
MANUAL_FPS = 60                 # e.g. 120.0; else use metadata

# SPEED / QUALITY TUNING
SCALE_FACTOR = 1                # 1.0 = full res. 0.5 = half (4x fewer pixels)
MAX_CORNERS = 50                # fewer features -> faster LK
REDETECT_EVERY_N_FRAMES = 30    # re-detect features every N frames
MIN_FEATURES_BEFORE_FORCE_REDETECT = 40
SKIP_FRAMES = 2                 # 1 = use every frame, 2 = use every other, etc.

# Prefer dark dots (blob-ish) instead of bright highlights
USE_DARK_DOT_MASK = True
DOT_BLUR_KSIZE = 7              # odd int; 5-9 typical
DOT_THRESH_METHOD = "otsu"      # "otsu" or "percentile"
DOT_DARK_PERCENTILE = 20        # used if method == "percentile"
DOT_MORPH_KSIZE = 5             # clean up blobs
DOT_MIN_AREA = 20               # reject tiny noise blobs (scaled px)
DOT_MAX_AREA = 5000             # reject huge regions (scaled px)
DOT_DILATE_KSIZE = 5            # expand dot mask so corners land on blobs
# =============================================================== #


def make_dark_dot_mask(gray, mask_ball):
    """
    Returns a binary mask (uint8 0/255) that highlights dark blobs inside the ball.
    Dark dots -> bright after inversion -> threshold -> clean -> area filter.
    """
    blur = cv2.GaussianBlur(gray, (DOT_BLUR_KSIZE, DOT_BLUR_KSIZE), 0)
    inv = 255 - blur

    if DOT_THRESH_METHOD.lower() == "otsu":
        _, bw = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        vals = inv[mask_ball > 0]
        if vals.size == 0:
            return np.zeros_like(gray, dtype=np.uint8)
        t = np.percentile(vals, 100 - DOT_DARK_PERCENTILE)
        _, bw = cv2.threshold(inv, t, 255, cv2.THRESH_BINARY)

    bw = cv2.bitwise_and(bw, mask_ball)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DOT_MORPH_KSIZE, DOT_MORPH_KSIZE))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(bw)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if DOT_MIN_AREA <= area <= DOT_MAX_AREA:
            out[labels == i] = 255

    if DOT_DILATE_KSIZE and DOT_DILATE_KSIZE > 1:
        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DOT_DILATE_KSIZE, DOT_DILATE_KSIZE))
        out = cv2.dilate(out, kd, iterations=1)

    return out


def load_ball_from_config(config_path: str, scale_factor: float):
    """
    Loads ball center/radius from config JSON saved by the calibration tool.

    Expected JSON fields:
      - ball_center_fullres: [cx, cy]
      - ball_radius_fullres: r

    Returns (cx_scaled, cy_scaled, r_scaled, cx_full, cy_full, r_full).
    """
    if not os.path.exists(config_path):
        raise RuntimeError(f"Ball config JSON not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "ball_center_fullres" not in cfg or "ball_radius_fullres" not in cfg:
        raise RuntimeError(
            "Config JSON missing required fields. "
            "Expected: ball_center_fullres, ball_radius_fullres"
        )

    cx_full, cy_full = cfg["ball_center_fullres"]
    r_full = cfg["ball_radius_fullres"]

    cx_scaled = int(round(float(cx_full) * scale_factor))
    cy_scaled = int(round(float(cy_full) * scale_factor))
    r_scaled = int(round(float(r_full) * scale_factor))

    return cx_scaled, cy_scaled, r_scaled, float(cx_full), float(cy_full), float(r_full)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame from video.")

    # Downscale first frame
    first_small = cv2.resize(
        first_frame,
        (0, 0),
        fx=SCALE_FACTOR,
        fy=SCALE_FACTOR,
        interpolation=cv2.INTER_AREA
    )
    gray_prev = cv2.cvtColor(first_small, cv2.COLOR_BGR2GRAY)

    # FPS
    fps_meta = cap.get(cv2.CAP_PROP_FPS)
    if MANUAL_FPS is not None:
        fps_raw = float(MANUAL_FPS)
    elif fps_meta and fps_meta > 1e-3:
        fps_raw = float(fps_meta)
    else:
        fps_raw = 60.0

    fps = fps_raw / SKIP_FRAMES
    print(f"Raw FPS: {fps_raw:.2f}, using every {SKIP_FRAMES} frame(s) -> effective FPS: {fps:.2f}")

    # Load ball geometry from config
    cx, cy, radius, cx_full, cy_full, r_full = load_ball_from_config(BALL_CONFIG_PATH, SCALE_FACTOR)
    print("Using ball geometry from config:")
    print(f"  Config file: {BALL_CONFIG_PATH}")
    print(f"  Center full-res: ({cx_full:.2f}, {cy_full:.2f}), radius full-res: {r_full:.2f} px")
    print(f"  Center scaled:   ({cx}, {cy}), radius scaled:   {radius} px")

    # Ball mask (scaled); OK if partially off-screen (OpenCV clips automatically)
    mask_ball = np.zeros_like(gray_prev, dtype=np.uint8)
    cv2.circle(mask_ball, (int(cx), int(cy)), int(radius), 255, -1)

    # Feature detection helper
    def detect_features(gray):
        if USE_DARK_DOT_MASK:
            dot_mask = make_dark_dot_mask(gray, mask_ball)
            use_mask = dot_mask
        else:
            dot_mask = None
            use_mask = mask_ball

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=MAX_CORNERS,
            qualityLevel=0.01,
            minDistance=5,
            mask=use_mask
        )
        return pts, dot_mask

    p0, dot_mask_prev = detect_features(gray_prev)
    if p0 is None:
        raise RuntimeError(
            "No features found within the ball mask. "
            "Try increasing MAX_CORNERS, improving contrast/lighting, or making dots more visible."
        )

    # Output CSV path
    video_stem = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    if OUTDIR:
        os.makedirs(OUTDIR, exist_ok=True)
        csv_path = os.path.join(OUTDIR, f"{video_stem}_ball_speed_fast.csv")
    else:
        base, _ = os.path.splitext(VIDEO_PATH)
        csv_path = base + "_ball_speed_fast.csv"

    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_idx",
        "time_sec",
        "omega_z_rad_per_s",
        "abs_omega_z_rad_per_s",
        "surface_speed_cm_per_s",
        "n_features_tracked",
        "n_features_used",
        "omega_std_rad_per_s"
    ])

    frame_idx = 0
    raw_idx = 0

    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_idx += 1

        if (raw_idx - 1) % SKIP_FRAMES != 0:
            continue

        frame_small = cv2.resize(
            frame,
            (0, 0),
            fx=SCALE_FACTOR,
            fy=SCALE_FACTOR,
            interpolation=cv2.INTER_AREA
        )
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, p0, None, **lk_params)

        if p1 is None or st is None:
            good_new = None
            good_old = None
            omega_samples = []
            n_tracked = 0
            n_used = 0
        else:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            omega_samples = []
            for (x_new, y_new), (x_old, y_old) in zip(good_new, good_old):
                rx, ry = x_old - cx, y_old - cy
                r_norm = math.hypot(rx, ry)

                # Reject points too close to center (unstable) or outside ball radius (background leak)
                if r_norm < 5 or r_norm > radius:
                    continue

                dx, dy = x_new - x_old, y_new - y_old

                # Tangent direction at (rx, ry): (-ry, rx)
                tx, ty = -ry, rx
                t_norm = math.hypot(tx, ty)
                if t_norm == 0:
                    continue
                tx /= t_norm
                ty /= t_norm

                v_tangential = dx * tx + dy * ty
                omega_z = v_tangential / r_norm
                omega_samples.append(omega_z)

            n_tracked = len(good_new)
            n_used = len(omega_samples)

        if omega_samples:
            omega_z_mean = float(np.median(omega_samples))
            omega_std = float(np.std(omega_samples))
            omega_z_rad_per_s = omega_z_mean * fps
            omega_std_rad_per_s = omega_std * fps
        else:
            omega_z_rad_per_s = 0.0
            omega_std_rad_per_s = 0.0

        abs_omega = abs(omega_z_rad_per_s)
        surface_speed_cm_per_s = abs_omega * BALL_RADIUS_CM

        time_sec = frame_idx / fps
        csv_writer.writerow([
            frame_idx,
            f"{time_sec:.6f}",
            f"{omega_z_rad_per_s:.6f}",
            f"{abs_omega:.6f}",
            f"{surface_speed_cm_per_s:.6f}",
            n_tracked,
            n_used,
            f"{omega_std_rad_per_s:.6f}"
        ])

        if SHOW_DEBUG:
            debug_frame = frame_small.copy()

            if USE_DARK_DOT_MASK and dot_mask_prev is not None:
                overlay = debug_frame.copy()
                overlay[dot_mask_prev > 0] = (255, 0, 0)
                debug_frame = cv2.addWeighted(debug_frame, 0.75, overlay, 0.25, 0)

            cv2.circle(debug_frame, (cx, cy), radius, (0, 0, 255), 2)

            if good_new is not None and good_old is not None:
                for (x_new, y_new), (x_old, y_old) in zip(good_new, good_old):
                    cv2.line(debug_frame,
                             (int(x_old), int(y_old)),
                             (int(x_new), int(y_new)),
                             (0, 255, 0), 1)
                    cv2.circle(debug_frame, (int(x_new), int(y_new)), 2, (0, 255, 0), -1)

            text = f"ωz ≈ {omega_z_rad_per_s:7.2f} rad/s  |  speed ≈ {surface_speed_cm_per_s:6.1f} cm/s"
            cv2.putText(debug_frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Ball tracking (scaled)", debug_frame)
            if cv2.waitKey(1) == 27:
                break

        # Re-detect features occasionally
        need_redetect = (
            frame_idx % REDETECT_EVERY_N_FRAMES == 0 or
            len(p0) < MIN_FEATURES_BEFORE_FORCE_REDETECT
        )

        if need_redetect:
            new_p0, dot_mask_prev = detect_features(gray)
            if new_p0 is not None:
                p0 = new_p0
            else:
                if good_new is not None:
                    p0 = good_new.reshape(-1, 1, 2)
        else:
            if good_new is not None:
                p0 = good_new.reshape(-1, 1, 2)

        gray_prev = gray
        frame_idx += 1

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print(f"Done. Processed {frame_idx} frames (from {raw_idx} raw frames).")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
