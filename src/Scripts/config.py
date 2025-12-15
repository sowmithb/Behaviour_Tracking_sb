import cv2
import numpy as np
import json
import os

# ===================== USER SETTINGS ===================== #
VIDEO_PATH = r"C:\Users\16083\Coding\Behaviour_Tracking\data\raw\Het 230\post.mp4"
OUTPUT_CONFIG_PATH = r"C:\Users\16083\Coding\Behaviour_Tracking_Self\data\proccessed\ball_config.json"
FRAME_INDEX = 60            # which frame to calibrate on (0 = first)
MAX_POINTS = 12            # more points = better fit; 6-10 is usually plenty
# ========================================================= #


def fit_circle_least_squares(points_xy):
    """
    Fits a circle to 2D points using a simple least-squares method.
    Returns (cx, cy, r).
    Works well even when the arc is partial (ball off-screen), as long as
    points span a decent portion of the outline.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]

    # Solve: x^2 + y^2 + D x + E y + F = 0
    # => [x y 1] [D E F]^T = -(x^2 + y^2)
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)

    # Least squares solution
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = sol

    cx = -D / 2.0
    cy = -E / 2.0
    r2 = cx * cx + cy * cy - F
    r = np.sqrt(max(r2, 0.0))
    return float(cx), float(cy), float(r)


def grab_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx}")
    return frame


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frame = grab_frame(cap, FRAME_INDEX)
    cap.release()

    h, w = frame.shape[:2]

    points = []
    dragging = {"down": False}

    window = "Calibrate Ball: click outline points (ENTER=save, R=reset, U=undo, ESC=quit)"

    def redraw(img):
        vis = img.copy()

        # draw clicked points
        for p in points:
            cv2.circle(vis, p, 4, (0, 255, 0), -1)

        # if enough points, show fitted circle preview
        if len(points) >= 3:
            cx, cy, r = fit_circle_least_squares(points)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)), (0, 0, 255), 2)
            cv2.circle(vis, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)

            txt = f"Fit: cx={cx:.1f}, cy={cy:.1f}, r={r:.1f}   points={len(points)}"
            cv2.putText(vis, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        help_txt = "L-click: add point | U: undo | R: reset | ENTER: save | ESC: exit"
        cv2.putText(vis, help_txt, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return vis

    def on_mouse(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < MAX_POINTS:
                points.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        vis = redraw(frame)
        cv2.imshow(window, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            print("Exit without saving.")
            break
        elif key in (13, 10):  # ENTER
            if len(points) < 3:
                print("Need at least 3 points to fit a circle.")
                continue

            cx, cy, r = fit_circle_least_squares(points)

            cfg = {
                "video_path": VIDEO_PATH,
                "frame_index": FRAME_INDEX,
                "image_width": w,
                "image_height": h,
                "ball_center_fullres": [cx, cy],
                "ball_radius_fullres": r,
                "n_points": len(points),
                "points_xy": points
            }

            if OUTPUT_CONFIG_PATH is None:
                base, _ = os.path.splitext(VIDEO_PATH)
                out_path = base + "_ball_config.json"
            else:
                out_path = OUTPUT_CONFIG_PATH

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            print(f"Saved ball config to: {out_path}")
            print(f"cx={cx:.3f}, cy={cy:.3f}, r={r:.3f} (full-res pixels)")
            break
        elif key in (ord('r'), ord('R')):
            points = []
        elif key in (ord('u'), ord('U')):
            if points:
                points.pop()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
