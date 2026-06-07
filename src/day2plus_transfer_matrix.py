"""
Day2+ Cross-Board Transfer Stress Test utilities.

This module is intentionally self-contained so the notebook stays concise.
It assumes UCI #361 raw text filenames follow the Day1/Day2 convention:
    B1_GMe_F010_R1.txt
where board=B1, gas=GMe, concentration_code=F010, replicate=R1.

No concentration-derived columns are used as input features.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGBOOST = True
except Exception:  # pragma: no cover
    XGBRegressor = None
    HAS_XGBOOST = False


FILENAME_RE = re.compile(
    r"(?P<board>B\d+)_(?P<gas>G[A-Za-z0-9]+)_(?P<conc>F\d+)_(?P<rep>R\d+)",
    re.IGNORECASE,
)

TARGET_GAS = "GMe"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    figures_dir: Path
    results_dir: Path
    gas: str = TARGET_GAS
    random_state: int = RANDOM_STATE

    def ensure_dirs(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


def parse_filename(path: Path) -> Optional[dict]:
    match = FILENAME_RE.search(path.stem)
    if not match:
        return None
    gd = match.groupdict()
    concentration_numeric = int(re.sub(r"\D", "", gd["conc"]))
    return {
        "filename": path.name,
        "sample_id": path.stem,
        "board": gd["board"].upper(),
        "gas": gd["gas"],
        "concentration_code": gd["conc"].upper(),
        "concentration_numeric": concentration_numeric,
        "replicate": gd["rep"].upper(),
    }


def _read_numeric_txt(path: Path) -> np.ndarray:
    """Read a txt file robustly as a numeric 2D array."""
    try:
        arr = np.loadtxt(path)
    except Exception:
        arr = np.genfromtxt(path, delimiter=None)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = arr[:, np.isfinite(arr).all(axis=0)] if arr.shape[1] > 1 else arr
    if arr.size == 0:
        raise ValueError(f"No numeric data found in {path}")
    return arr


def _sensor_matrix(arr: np.ndarray) -> np.ndarray:
    """Remove an obvious time/index column when present and return sensor signals."""
    if arr.shape[1] <= 1:
        return arr
    first = arr[:, 0]
    diffs = np.diff(first)
    monotonic = np.all(diffs >= 0) or np.all(diffs <= 0)
    # UCI files often include time/index as first column. Drop if monotonic and unlike sensors.
    if monotonic and np.nanstd(first) > 0:
        return arr[:, 1:]
    return arr


def _safe_slope(y: np.ndarray) -> float:
    if len(y) < 3 or np.nanstd(y) == 0:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def extract_features_from_signal(signal: np.ndarray) -> Tuple[dict, dict, dict]:
    """Return raw, physics-informed, and combined feature dictionaries."""
    X = _sensor_matrix(signal)
    n, m = X.shape
    if n < 5:
        raise ValueError("Signal is too short for feature extraction")

    baseline_n = max(3, int(0.10 * n))
    early_n = max(3, int(0.25 * n))
    late_n = max(3, int(0.25 * n))

    baseline = np.nanmedian(X[:baseline_n, :], axis=0)
    early = X[:early_n, :]
    late = X[-late_n:, :]
    peak = np.nanmax(X, axis=0)
    trough = np.nanmin(X, axis=0)
    final = np.nanmedian(late, axis=0)

    eps = 1e-12
    response_pos = peak - baseline
    response_neg = baseline - trough
    response_abs = np.where(np.abs(response_pos) >= np.abs(response_neg), response_pos, -response_neg)
    sensitivity_ratio = (baseline + response_abs + eps) / (baseline + eps)
    recovery_residual = final - baseline

    raw = {}
    physics = {}

    for j in range(m):
        col = X[:, j]
        raw.update({
            f"raw_s{j+1}_mean": float(np.nanmean(col)),
            f"raw_s{j+1}_std": float(np.nanstd(col)),
            f"raw_s{j+1}_min": float(np.nanmin(col)),
            f"raw_s{j+1}_max": float(np.nanmax(col)),
            f"raw_s{j+1}_q25": float(np.nanpercentile(col, 25)),
            f"raw_s{j+1}_q75": float(np.nanpercentile(col, 75)),
        })
        physics.update({
            f"phys_s{j+1}_baseline": float(baseline[j]),
            f"phys_s{j+1}_response_abs": float(response_abs[j]),
            f"phys_s{j+1}_response_norm": float(response_abs[j] / (abs(baseline[j]) + eps)),
            f"phys_s{j+1}_sensitivity_ratio": float(sensitivity_ratio[j]),
            f"phys_s{j+1}_early_slope": _safe_slope(early[:, j]),
            f"phys_s{j+1}_late_slope": _safe_slope(late[:, j]),
            f"phys_s{j+1}_recovery_residual": float(recovery_residual[j]),
            f"phys_s{j+1}_recovery_norm": float(recovery_residual[j] / (abs(response_abs[j]) + eps)),
        })

    # Cross-sensor population descriptors: useful for robustness geometry.
    physics.update({
        "phys_response_l2": float(np.linalg.norm(response_abs)),
        "phys_response_mean": float(np.nanmean(response_abs)),
        "phys_response_std": float(np.nanstd(response_abs)),
        "phys_baseline_mean": float(np.nanmean(baseline)),
        "phys_baseline_std": float(np.nanstd(baseline)),
        "phys_recovery_l2": float(np.linalg.norm(recovery_residual)),
        "phys_recovery_mean_abs": float(np.nanmean(np.abs(recovery_residual))),
    })
    combined = {**raw, **physics}
    return raw, physics, combined


def build_feature_table(data_dir: Path, gas: str = TARGET_GAS) -> pd.DataFrame:
    rows: List[dict] = []
    files = sorted(Path(data_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    for path in files:
        meta = parse_filename(path)
        if meta is None or meta["gas"] != gas:
            continue
        signal = _read_numeric_txt(path)
        raw, physics, combined = extract_features_from_signal(signal)
        rows.append({**meta, **combined})

    if not rows:
        raise ValueError(f"No gas={gas} files found in {data_dir}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["board", "concentration_numeric", "replicate", "filename"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    exclude = {
        "filename", "sample_id", "board", "gas", "concentration_code",
        "concentration_numeric", "replicate", "source_boards", "target_board",
    }
    if feature_set == "raw":
        cols = [c for c in df.columns if c.startswith("raw_")]
    elif feature_set == "physics":
        cols = [c for c in df.columns if c.startswith("phys_")]
    elif feature_set == "combined":
        cols = [c for c in df.columns if c not in exclude and (c.startswith("raw_") or c.startswith("phys_"))]
    else:
        raise ValueError("feature_set must be one of: raw, physics, combined")
    if not cols:
        raise ValueError(f"No columns found for feature_set={feature_set}")
    return cols


def make_model(name: str, random_state: int = RANDOM_STATE):
    if name == "LinearRegression":
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "XGBoost":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed; skip model='XGBoost' or install xgboost")
        return XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def evaluate_split(
    df: pd.DataFrame,
    source_boards: Sequence[str],
    target_board: str,
    feature_set: str = "physics",
    model_name: str = "RandomForest",
    random_state: int = RANDOM_STATE,
) -> Tuple[dict, pd.DataFrame]:
    source_boards = tuple(sorted({b.upper() for b in source_boards}))
    target_board = target_board.upper()
    if target_board in source_boards:
        raise ValueError("target_board must not be included in source_boards")

    train_df = df[df["board"].isin(source_boards)].copy()
    test_df = df[df["board"].eq(target_board)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"Empty train/test split: source={source_boards}, target={target_board}")

    feature_cols = get_feature_columns(df, feature_set)
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train_df["concentration_numeric"].astype(float)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_test = test_df["concentration_numeric"].astype(float)

    model = make_model(model_name, random_state)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = math.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred) if len(np.unique(y_test)) > 1 else np.nan

    pred_df = test_df[["filename", "sample_id", "board", "concentration_code", "concentration_numeric", "replicate"]].copy()
    pred_df["source_boards"] = "+".join(source_boards)
    pred_df["target_board"] = target_board
    pred_df["feature_set"] = feature_set
    pred_df["model"] = model_name
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = pred
    pred_df["abs_error"] = np.abs(pred_df["y_true"] - pred_df["y_pred"])

    metrics = {
        "source_boards": "+".join(source_boards),
        "n_source_boards": len(source_boards),
        "target_board": target_board,
        "feature_set": feature_set,
        "model": model_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "includes_B5_source": "B5" in source_boards,
    }
    return metrics, pred_df


def run_single_source_matrix(
    df: pd.DataFrame,
    feature_sets: Sequence[str] = ("raw", "physics", "combined"),
    model_names: Sequence[str] = ("LinearRegression", "RandomForest", "XGBoost"),
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    boards = sorted(df["board"].unique())
    metrics_rows: List[dict] = []
    pred_rows: List[pd.DataFrame] = []
    for feature_set, model_name, source, target in itertools.product(feature_sets, model_names, boards, boards):
        if source == target:
            continue
        try:
            metrics, preds = evaluate_split(df, [source], target, feature_set, model_name, random_state)
        except ImportError as exc:
            warnings.warn(str(exc))
            continue
        metrics_rows.append(metrics)
        pred_rows.append(preds)
    return pd.DataFrame(metrics_rows), pd.concat(pred_rows, ignore_index=True)


def run_leave_one_out(
    df: pd.DataFrame,
    feature_sets: Sequence[str] = ("raw", "physics", "combined"),
    model_names: Sequence[str] = ("LinearRegression", "RandomForest", "XGBoost"),
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    boards = sorted(df["board"].unique())
    metrics_rows: List[dict] = []
    pred_rows: List[pd.DataFrame] = []
    for feature_set, model_name, target in itertools.product(feature_sets, model_names, boards):
        sources = [b for b in boards if b != target]
        try:
            metrics, preds = evaluate_split(df, sources, target, feature_set, model_name, random_state)
        except ImportError as exc:
            warnings.warn(str(exc))
            continue
        metrics_rows.append(metrics)
        pred_rows.append(preds)
    return pd.DataFrame(metrics_rows), pd.concat(pred_rows, ignore_index=True)


def run_mixed_domain_combinations(
    df: pd.DataFrame,
    source_sizes: Sequence[int] = (2, 3, 4),
    feature_set: str = "physics",
    model_name: str = "RandomForest",
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    boards = sorted(df["board"].unique())
    metrics_rows: List[dict] = []
    pred_rows: List[pd.DataFrame] = []
    for target in boards:
        candidates = [b for b in boards if b != target]
        for k in source_sizes:
            if k > len(candidates):
                continue
            for sources in itertools.combinations(candidates, k):
                metrics, preds = evaluate_split(df, sources, target, feature_set, model_name, random_state)
                metrics_rows.append(metrics)
                pred_rows.append(preds)
    return pd.DataFrame(metrics_rows), pd.concat(pred_rows, ignore_index=True)


def make_transfer_matrix(metrics_df: pd.DataFrame, metric: str, feature_set: str, model: str) -> pd.DataFrame:
    tmp = metrics_df[(metrics_df["feature_set"] == feature_set) & (metrics_df["model"] == model)].copy()
    tmp = tmp[tmp["n_source_boards"] == 1]
    return tmp.pivot(index="source_boards", columns="target_board", values=metric).sort_index().sort_index(axis=1)


def plot_heatmap(matrix: pd.DataFrame, title: str, path: Path, cmap: str = "viridis", fmt: str = ".2f") -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=160)
    im = ax.imshow(matrix.values.astype(float), aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Target board")
    ax.set_ylabel("Source board")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, format(float(val), fmt), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def compute_asymmetry(metrics_df: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    single = metrics_df[metrics_df["n_source_boards"] == 1].copy()
    rows = []
    for _, row in single.iterrows():
        src = row["source_boards"]
        tgt = row["target_board"]
        reverse = single[
            (single["source_boards"] == tgt)
            & (single["target_board"] == src)
            & (single["feature_set"] == row["feature_set"])
            & (single["model"] == row["model"])
        ]
        if reverse.empty or src > tgt:
            continue
        rev_val = float(reverse.iloc[0][metric])
        val = float(row[metric])
        rows.append({
            "board_pair": f"{src}<->{tgt}",
            "source_to_target": f"{src}->{tgt}",
            "reverse": f"{tgt}->{src}",
            "feature_set": row["feature_set"],
            "model": row["model"],
            f"{metric}_forward": val,
            f"{metric}_reverse": rev_val,
            f"{metric}_abs_asymmetry": abs(val - rev_val),
            f"{metric}_signed_asymmetry": val - rev_val,
        })
    return pd.DataFrame(rows)


def plot_asymmetry(asym_df: pd.DataFrame, feature_set: str, model: str, path: Path, metric: str = "rmse") -> None:
    tmp = asym_df[(asym_df["feature_set"] == feature_set) & (asym_df["model"] == model)].copy()
    tmp = tmp.sort_values(f"{metric}_abs_asymmetry", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.bar(tmp["board_pair"], tmp[f"{metric}_abs_asymmetry"])
    ax.set_ylabel(f"|Δ {metric.upper()}|")
    ax.set_xlabel("Board pair")
    ax.set_title(f"Transfer asymmetry — {model}, {feature_set}")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_pca_by_board_role(df: pd.DataFrame, source_boards: Sequence[str], target_board: str, feature_set: str, path: Path) -> pd.DataFrame:
    cols = get_feature_columns(df, feature_set)
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Xs = StandardScaler().fit_transform(X)
    emb = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)
    pca_df = df[["board", "concentration_numeric", "sample_id"]].copy()
    pca_df["PC1"] = emb[:, 0]
    pca_df["PC2"] = emb[:, 1]
    pca_df["role"] = np.where(pca_df["board"].isin(source_boards), "source", np.where(pca_df["board"].eq(target_board), "target", "unused"))

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=160)
    markers = {"source": "o", "target": "X", "unused": "."}
    for role, part in pca_df.groupby("role"):
        ax.scatter(part["PC1"], part["PC2"], label=role, marker=markers.get(role, "o"), alpha=0.8)
    for board, part in pca_df.groupby("board"):
        ax.text(part["PC1"].mean(), part["PC2"].mean(), board, fontsize=10, weight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA geometry: source={'+'.join(source_boards)}, target={target_board}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return pca_df


def plot_concentration_failure(pred_df: pd.DataFrame, feature_set: str, model: str, path: Path) -> pd.DataFrame:
    tmp = pred_df[(pred_df["feature_set"] == feature_set) & (pred_df["model"] == model)].copy()
    summary = tmp.groupby(["source_boards", "target_board", "concentration_numeric"], as_index=False)["abs_error"].mean()
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=160)
    for target, part in summary.groupby("target_board"):
        target_part = part.groupby("concentration_numeric", as_index=False)["abs_error"].mean()
        ax.plot(target_part["concentration_numeric"], target_part["abs_error"], marker="o", label=target)
    ax.set_xlabel("Methane concentration code value")
    ax.set_ylabel("Mean absolute error")
    ax.set_title(f"Concentration-dependent transfer failure — {model}, {feature_set}")
    ax.legend(title="Target")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return summary


def plot_source_diversity(metrics_df: pd.DataFrame, path: Path, feature_set: str = "physics", model: str = "RandomForest") -> pd.DataFrame:
    tmp = metrics_df[(metrics_df["feature_set"] == feature_set) & (metrics_df["model"] == model)].copy()
    summary = tmp.groupby(["n_source_boards", "includes_B5_source"], as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"), mae_mean=("mae", "mean"), r2_mean=("r2", "mean")
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    for inc, part in summary.groupby("includes_B5_source"):
        ax.errorbar(part["n_source_boards"], part["rmse_mean"], yerr=part["rmse_std"].fillna(0), marker="o", label=f"B5 source={inc}")
    ax.set_xlabel("Number of source boards")
    ax.set_ylabel("Mean RMSE across targets")
    ax.set_title("Source-domain diversity and drift-inclusive robustness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return summary


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def assert_no_label_leakage(df: pd.DataFrame) -> dict:
    leakage_names = {"concentration", "concentration_numeric", "concentration_code", "target", "y_true", "y_pred"}
    feature_cols = {fs: set(get_feature_columns(df, fs)) for fs in ["raw", "physics", "combined"]}
    report = {}
    for fs, cols in feature_cols.items():
        overlap = sorted(cols & leakage_names)
        report[fs] = {"n_features": len(cols), "leakage_overlap": overlap, "ok": len(overlap) == 0}
        if overlap:
            raise AssertionError(f"Potential label leakage in {fs}: {overlap}")
    return report
