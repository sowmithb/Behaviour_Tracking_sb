# Repository Summary: Behaviour Ball-Speed Tracking

This repository contains Python scripts for estimating mouse locomotor behaviour from videos of an enclosed behavioural setup with a running ball. The main output is a per-frame estimate of ball angular velocity and surface speed, followed by summary reports that quantify movement above or below a configurable speed threshold.

## What The Code Does

The workflow tracks visual features on the surface of a ball in video recordings. It uses the calibrated ball center and radius to convert optical-flow motion of surface dots into angular velocity around the image-plane z-axis, then converts angular speed into surface speed in cm/s using the configured physical ball radius.

The intended pipeline is:

1. Calibrate the visible ball outline and save `src/ball_config.json`.
2. Run tracking on one or more raw `.mp4` videos.
3. Save per-video speed CSVs under `data/proccessed`.
4. Generate PDF reports and movement-bout CSVs under `data/reports`.

## Repository Layout

```text
.
├── requirements.txt
├── REPOSITORY_SUMMARY.md
├── data/
│   ├── raw/              # Expected input videos
│   ├── proccessed/       # Tracking CSV outputs; note spelling in code
│   └── reports/          # PDF reports and bout CSV outputs
└── src/
    ├── ball_config.json
    ├── Pipe/
    │   ├── run_tracking_pipeline.py
    │   └── run_report_pipeline.py
    └── Scripts/
        ├── config.py
        ├── track_ball.py
        ├── report_from_csv.py
        └── plot_one.py
```

## Main Scripts

### `src/Scripts/config.py`

Interactive calibration script. It opens a selected video frame, lets the user click points along the ball outline, fits a circle by least squares, and writes the ball geometry to `src/ball_config.json`.

The config file stores:

- `ball_center_fullres`: full-resolution ball center `[cx, cy]`
- `ball_radius_fullres`: full-resolution ball radius in pixels
- frame metadata and clicked outline points

This calibration is important because tracking uses the ball center and radius to separate true ball-surface motion from background or unstable center-region motion.

### `src/Scripts/track_ball.py`

Core tracking script for a single video. It:

- Opens an input video supplied by `--video`.
- Loads ball geometry from `--config` or the default config path.
- Builds a circular mask for the ball.
- Detects dark surface dots/features on the ball.
- Tracks features frame-to-frame with OpenCV Lucas-Kanade optical flow.
- Projects each feature displacement onto the local tangent direction of the ball.
- Estimates median angular velocity from valid tracked features.
- Converts angular velocity to surface speed using `BALL_RADIUS_CM = 3.2`.
- Writes a `*_ball_speed_fast.csv` output.

CSV columns include:

- `frame_idx`
- `time_sec`
- `omega_z_rad_per_s`
- `abs_omega_z_rad_per_s`
- `surface_speed_cm_per_s`
- `n_features_tracked`
- `n_features_used`
- `omega_std_rad_per_s`

Important tracking parameters in the script include FPS, frame skipping, dark-dot thresholding, feature count, redetection frequency, and optical-flow settings.

### `src/Pipe/run_tracking_pipeline.py`

Batch runner for tracking all videos in the raw-data tree. It expects raw files in this structure:

```text
data/raw/<cohort>/<date>/<animal>/*.mp4
```

For each video, it calls `track_ball.py` and writes outputs while preserving the cohort/date/animal structure under:

```text
data/proccessed/<cohort>/<date>/<animal>/
```

It supports optional cohort filtering:

```bash
python src/Pipe/run_tracking_pipeline.py --cohort "Cohort 2"
```

### `src/Scripts/report_from_csv.py`

Report generator for one tracking CSV. It reads a `*_ball_speed_fast.csv`, smooths speed and angular speed, thresholds smoothed surface speed, computes summary metrics, and writes a report package.

Outputs for each CSV include:

- `report.pdf`
- `above_threshold_bouts.csv`
- `below_threshold_bouts.csv`

The report includes raw and smoothed plots for surface speed and angular speed, movement timing above/below threshold, bout counts/durations, speed summaries, distance proxy, and feature-quality summaries when available.

### `src/Pipe/run_report_pipeline.py`

Batch runner for generating reports from processed CSVs. It scans:

```text
data/proccessed/**/*_ball_speed_fast.csv
```

and mirrors the processed-folder structure under:

```text
data/reports/
```

Default report settings in the pipeline are:

- speed threshold: `0.5 cm/s`
- smoothing window: `0.5 s`

It also supports optional cohort filtering:

```bash
python src/Pipe/run_report_pipeline.py --cohort "Cohort 2"
```

### `src/Scripts/plot_one.py`

Quick visual inspection utility for one tracking CSV. It plots surface speed and angular speed against time, with optional smoothing and a speed-threshold line.

Example:

```bash
python src/Scripts/plot_one.py --csv path/to/video_ball_speed_fast.csv
```

## Dependencies

The repository depends on:

- `numpy`
- `pandas`
- `opencv-python`
- `matplotlib`
- `plotly`
- `tqdm`
- `playwright`
- `reportlab`

Install with:

```bash
pip install -r requirements.txt
```

## Data Flow

```text
raw video (.mp4)
    ↓
ball calibration (`config.py`)
    ↓
ball geometry (`src/ball_config.json`)
    ↓
tracking (`track_ball.py` or `run_tracking_pipeline.py`)
    ↓
per-frame speed CSV (`*_ball_speed_fast.csv`)
    ↓
reporting (`report_from_csv.py` or `run_report_pipeline.py`)
    ↓
PDF report + above/below threshold bout CSVs
```

## Key Assumptions And Notes

- The ball has visible surface features, especially dark dots, that can be tracked frame-to-frame.
- The tracking script is tuned for batch mode with debug display disabled.
- `MANUAL_FPS` is currently set to `60`, so the tracker uses 60 FPS rather than relying on video metadata.
- `SKIP_FRAMES = 2`, so effective sampling is 30 Hz when `MANUAL_FPS = 60`.
- The physical ball radius is hard-coded as `BALL_RADIUS_CM = 3.2`.
- The output folder is spelled `proccessed` throughout the code; changing this spelling would require updating both tracking and report pipeline paths.
- Several hard-coded Windows paths remain in `config.py`, `track_ball.py`, and `ball_config.json`. For portability, pass paths through CLI arguments where supported, especially `--video`, `--outdir`, and `--config`.

## Typical Commands

Run tracking for one video:

```bash
python src/Scripts/track_ball.py \
  --video data/raw/<cohort>/<date>/<animal>/<video>.mp4 \
  --outdir data/proccessed/<cohort>/<date>/<animal> \
  --config src/ball_config.json
```

Run batch tracking:

```bash
python src/Pipe/run_tracking_pipeline.py
```

Generate a report for one CSV:

```bash
python src/Scripts/report_from_csv.py \
  --csv data/proccessed/<cohort>/<date>/<animal>/<video>_ball_speed_fast.csv \
  --processed-root data/proccessed \
  --reports-root data/reports \
  --thresh 0.5 \
  --smooth-sec 0.5
```

Run batch report generation:

```bash
python src/Pipe/run_report_pipeline.py
```
