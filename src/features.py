"""Feature extraction for MOx calibration transfer experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def raw_summary_features(x: pd.DataFrame, sensor_cols: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for col in sensor_cols:
        s = x[col].astype(float).to_numpy()
        features[f"{col}_mean"] = float(np.nanmean(s))
        features[f"{col}_std"] = float(np.nanstd(s))
        features[f"{col}_min"] = float(np.nanmin(s))
        features[f"{col}_max"] = float(np.nanmax(s))
        features[f"{col}_last"] = float(s[-1])
    return features


def physics_informed_features(
    x: pd.DataFrame,
    sensor_cols: list[str],
    time_col: str = "time_s",
    baseline_window_s: tuple[float, float] = (0, 30),
    response_window_s: tuple[float, float] = (60, 300),
    recovery_window_s: tuple[float, float] = (300, 600),
) -> dict[str, float]:
    """Extract simple interpretable MOx response features.

    Assumption to verify on Day 1: each trial contains a baseline period followed
    by gas exposure and recovery. If the official timing differs, update the
    windows before modeling.
    """
    t = x[time_col].to_numpy()
    features: dict[str, float] = {}

    base_mask = (t >= baseline_window_s[0]) & (t <= baseline_window_s[1])
    resp_mask = (t >= response_window_s[0]) & (t <= response_window_s[1])
    rec_mask = (t >= recovery_window_s[0]) & (t <= recovery_window_s[1])

    for col in sensor_cols:
        y = x[col].astype(float).to_numpy()
        r0 = float(np.nanmedian(y[base_mask]))
        resp = y[resp_mask]
        rec = y[rec_mask]
        tt_resp = t[resp_mask]
        tt_rec = t[rec_mask]

        eps = 1e-12
        norm = y / (r0 + eps)
        delta = (y - r0) / (abs(r0) + eps)
        peak_idx = int(np.nanargmax(np.abs(delta)))
        peak_delta = float(delta[peak_idx])

        features[f"{col}_r0"] = r0
        features[f"{col}_rs_over_r0_final"] = float(np.nanmedian(y[-300:]) / (r0 + eps))
        features[f"{col}_norm_mean_resp"] = float(np.nanmean(norm[resp_mask]))
        features[f"{col}_delta_peak"] = peak_delta
        features[f"{col}_delta_auc_resp"] = float(np.trapz(delta[resp_mask], tt_resp)) if len(tt_resp) > 1 else np.nan
        features[f"{col}_deriv_mean_resp"] = float(np.nanmean(np.gradient(resp, tt_resp))) if len(resp) > 2 else np.nan
        features[f"{col}_deriv_mean_recovery"] = float(np.nanmean(np.gradient(rec, tt_rec))) if len(rec) > 2 else np.nan

        target90 = 0.9 * peak_delta
        if peak_delta >= 0:
            crossed = np.where(delta >= target90)[0]
        else:
            crossed = np.where(delta <= target90)[0]
        features[f"{col}_t90_s"] = float(t[crossed[0]]) if len(crossed) else np.nan

    return features
