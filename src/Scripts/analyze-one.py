#!/usr/bin/env python3
"""
Generate an interactive (Plotly) HTML report from a single tracking CSV.

Usage:
  python report_interactive_from_csv.py --csv path/to/file.csv
  python report_interactive_from_csv.py --csv path/to/file.csv --fps 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------------------------------
# Column detection helpers
# -------------------------------------------------

def pick_time_column(df: pd.DataFrame, fps: Optional[float]) -> Tuple[pd.Series, str]:
    cols = {c.lower(): c for c in df.columns}

    for key in ["time_s", "t_s", "time", "timestamp", "seconds", "sec"]:
        if key in cols:
            return pd.to_numeric(df[cols[key]], errors="coerce"), cols[key]

    if "frame" in cols and fps:
        return pd.to_numeric(df[cols["frame"]], errors="coerce") / fps, "frame / fps"

    if fps:
        t = np.arange(len(df), dtype=float) / fps
        return pd.Series(t), "index / fps"

    return pd.Series(np.arange(len(df), dtype=float)), "index"


def pick_speed_column(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    cols = {c.lower(): c for c in df.columns}

    for key in [
        "speed", "speed_mm_s", "speed_cm_s", "speed_m_s",
        "ball_speed", "linear_speed", "velocity"
    ]:
        if key in cols:
            return pd.to_numeric(df[cols[key]], errors="coerce"), cols[key]

    dx = next((cols[k] for k in cols if k in ["dx", "delta_x"]), None)
    dy = next((cols[k] for k in cols if k in ["dy", "delta_y"]), None)

    if dx and dy:
        disp = np.sqrt(
            pd.to_numeric(df[dx], errors="coerce") ** 2 +
            pd.to_numeric(df[dy], errors="coerce") ** 2
        )
        return disp, f"sqrt({dx}² + {dy}²)"

    return pd.Series(np.nan, index=df.index), "speed (not found)"


def rolling_mean(series: pd.Series, window_s: float, fps: Optional[float], t: pd.Series):
    if fps:
        w = max(1, int(window_s * fps))
        return series.rolling(w, min_periods=max(1, w // 4)).mean()

    dt = np.nanmedian(np.diff(t.dropna()))
    if not np.isfinite(dt):
        return series

    w = max(1, int(window_s / dt))
    return series.rolling(w, min_periods=max(1, w // 4)).mean()


# -------------------------------------------------
# Report generation
# -------------------------------------------------

def build_report(
    df: pd.DataFrame,
    csv_path: Path,
    out_html: Path,
    fps: Optional[float],
    rolling_window_s: float,
):
    t, t_label = pick_time_column(df, fps)
    speed, speed_label = pick_speed_column(df)

    valid = t.notna()
    t = t[valid].reset_index(drop=True)
    speed = speed[valid].reset_index(drop=True)

    speed_roll = rolling_mean(speed, rolling_window_s, fps, t)

    dt = np.diff(t.values, prepend=t.values[0])
    cum_dist = np.nancumsum(speed.values * dt)

    stats = {
        "Rows": len(df),
        "Time source": t_label,
        "Speed source": speed_label,
        "Mean speed": np.nanmean(speed),
        "Median speed": np.nanmedian(speed),
        "95th percentile": np.nanpercentile(speed, 95),
        "Max speed": np.nanmax(speed),
        "FPS (arg)": fps if fps else "n/a",
    }

    fig = make_subplots(
        rows=3,
        cols=2,
        column_widths=[0.7, 0.3],
        row_heights=[0.45, 0.35, 0.2],
        specs=[
            [{"type": "scatter"}, {"type": "table"}],
            [{"type": "scatter"}, {"type": "histogram"}],
            [{"type": "scatter", "colspan": 2}, None],
        ],
        subplot_titles=[
            "Speed vs Time",
            "Summary",
            "Rolling Speed",
            "Speed Distribution",
            "Cumulative Distance (approx)",
        ],
    )

    fig.add_trace(
        go.Scatter(x=t, y=speed, mode="lines", name="Speed (raw)"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=t, y=speed_roll, mode="lines", name=f"Rolling ({rolling_window_s}s)"),
        row=1, col=1,
    )

    fig.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"]),
            cells=dict(
                values=[
                    list(stats.keys()),
                    [f"{v:.4g}" if isinstance(v, float) else v for v in stats.values()],
                ]
            ),
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Scatter(x=t, y=speed_roll, mode="lines", name="Rolling speed"),
        row=2, col=1,
    )

    fig.add_trace(
        go.Histogram(x=speed.dropna(), nbinsx=60),
        row=2, col=2,
    )

    fig.add_trace(
        go.Scatter(x=t, y=cum_dist, mode="lines", name="Cumulative distance"),
        row=3, col=1,
    )

    fig.update_layout(
        title=f"Interactive Tracking Report — {csv_path.name}",
        height=900,
        hovermode="x unified",
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] Wrote interactive report: {out_html}")


# -------------------------------------------------
# CLI
# -------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to tracking CSV")
    ap.add_argument("--fps", type=float, default=None, help="FPS if CSV lacks time column")
    ap.add_argument("--rolling-window-s", type=float, default=0.25)
    ap.add_argument("--out", default=None, help="Output HTML path")

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    out_html = (
        Path(args.out).expanduser().resolve()
        if args.out
        else csv_path.parent / f"{csv_path.stem}_interactive.html"
    )


    df = pd.read_csv(csv_path)

    build_report(
        df=df,
        csv_path=csv_path,
        out_html=out_html,
        fps=args.fps,
        rolling_window_s=args.rolling_window_s,
    )


if __name__ == "__main__":
    main()
