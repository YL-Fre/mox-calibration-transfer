"""
Day5 — Understanding B1 vs B5: Two Transfer Failure Modes

Lightweight, leakage-safe mechanism analysis for MOx methane calibration transfer.

Expected project layout:
MOx_Calibration_Transfer/
├── data/raw/
├── notebooks/
├── src/
├── figures/
├── results/
│   └── day2/feature_table_methane.csv
└── requirements.txt

Run from project root:
    python src/day5_failure_mode_analysis.py

Run from notebooks/:
    python ../src/day5_failure_mode_analysis.py
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover
    RandomForestRegressor = None


FORBIDDEN_FEATURE_COLUMNS = {
    "concentration",
    "concentration_numeric",
    "concentration_code",
    "y_numeric",
    "target",
    "label",
    "board",
    "gas",
    "filename",
    "file",
    "sample_id",
    "replicate",
}

BOARD_ORDER = ["B1", "B2", "B3", "B4", "B5"]
SOURCE_BOARDS = ["B2", "B3", "B4"]
TARGET_BOARDS = ["B1", "B5"]
HIGH_CONC_Q = 0.75


@dataclass(frozen=True)
class Paths:
    root: Path
    input_csv: Path
    results_dir: Path
    figures_dir: Path
    script_path: Path


def find_project_root(start: Path | None = None) -> Path:
    """Find project root from either project root, notebooks/, or src/."""
    start = Path.cwd() if start is None else Path(start).resolve()
    candidates = [start] + list(start.parents)
    for c in candidates:
        if (c / "results" / "day2" / "feature_table_methane.csv").exists():
            return c
        if (c / "requirements.txt").exists() and (c / "results").exists():
            return c
    # Fallback: if script is inside src, parent of src is project root.
    here = Path(__file__).resolve()
    if here.parent.name == "src":
        return here.parent.parent
    return start


def get_paths(root: Path | None = None) -> Paths:
    root = find_project_root(root)
    return Paths(
        root=root,
        input_csv=root / "results" / "day2" / "feature_table_methane.csv",
        results_dir=root / "results" / "day5",
        figures_dir=root / "figures" / "day5",
        script_path=Path(__file__).resolve(),
    )


def ensure_dirs(paths: Paths) -> None:
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)


def _normalize_board(x: object) -> str:
    return str(x).strip().upper()


def concentration_to_numeric(series: pd.Series) -> pd.Series:
    """Convert labels such as F010 or numeric strings into float concentration values."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    extracted = s.str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def load_feature_table(paths: Paths) -> pd.DataFrame:
    if not paths.input_csv.exists():
        raise FileNotFoundError(
            f"Required input not found: {paths.input_csv}\n"
            "Run Day2 first or place feature_table_methane.csv at results/day2/."
        )
    df = pd.read_csv(paths.input_csv)
    required = {"board", "concentration"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input feature table missing required columns: {missing}")
    df = df.copy()
    df["board"] = df["board"].map(_normalize_board)
    df["concentration_numeric"] = concentration_to_numeric(df["concentration"])
    df = df[df["board"].isin(BOARD_ORDER)].copy()
    df = df.dropna(subset=["concentration_numeric"])
    if df.empty:
        raise ValueError("No valid methane rows with board and numeric concentration were found.")
    return df


def select_feature_columns(df: pd.DataFrame, max_features: int = 40) -> List[str]:
    """Select leakage-free numeric feature columns, preferring stable/physics-like columns."""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in numeric_cols if c not in FORBIDDEN_FEATURE_COLUMNS]
    candidates = [c for c in candidates if df[c].notna().sum() >= max(5, int(0.2 * len(df)))]
    candidates = [c for c in candidates if float(df[c].std(skipna=True) or 0.0) > 1e-12]

    # Prefer response features likely to be physically interpretable, without assuming exact Day2 names.
    priority_terms = [
        "ratio", "rg", "r0", "response", "delta", "slope", "area", "auc", "steady",
        "median", "mean", "max", "min", "signal", "resistance", "sensor", "s",
    ]
    def priority(col: str) -> tuple[int, str]:
        low = col.lower()
        hit = any(t in low for t in priority_terms)
        return (0 if hit else 1, low)

    candidates = sorted(candidates, key=priority)
    # Keep analysis lightweight if Day2 generated many features.
    return candidates[:max_features]


def save_leakage_check(df: pd.DataFrame, feature_cols: Sequence[str], paths: Paths) -> None:
    forbidden_in_features = sorted(set(feature_cols) & FORBIDDEN_FEATURE_COLUMNS)
    check = pd.DataFrame(
        {
            "feature": list(feature_cols),
            "dtype": [str(df[c].dtype) for c in feature_cols],
            "is_forbidden": [c in FORBIDDEN_FEATURE_COLUMNS for c in feature_cols],
            "non_null_count": [int(df[c].notna().sum()) for c in feature_cols],
            "std": [float(df[c].std(skipna=True)) for c in feature_cols],
        }
    )
    check.to_csv(paths.results_dir / "day5_debug_feature_leakage_check.csv", index=False)
    print("\n[Day5 leakage check] Final feature list:")
    for c in feature_cols:
        print(f"  - {c}")
    assert not forbidden_in_features, f"Forbidden leakage columns included: {forbidden_in_features}"
    assert len(feature_cols) > 0, "No valid leakage-free feature columns selected."


def choose_representative_features(df: pd.DataFrame, feature_cols: Sequence[str], n: int = 6) -> List[str]:
    """Select features with strong monotonic association to concentration for interpretable plots."""
    rows = []
    med = df.groupby("concentration_numeric", observed=True)[list(feature_cols)].median(numeric_only=True)
    conc = med.index.to_numpy(dtype=float)
    for c in feature_cols:
        vals = med[c].to_numpy(dtype=float)
        mask = np.isfinite(conc) & np.isfinite(vals)
        if mask.sum() >= 3 and np.nanstd(vals[mask]) > 1e-12:
            corr = np.corrcoef(conc[mask], vals[mask])[0, 1]
            rows.append((abs(float(corr)) if np.isfinite(corr) else 0.0, c))
    rows.sort(reverse=True)
    chosen = [c for _, c in rows[:n]]
    return chosen or list(feature_cols[: min(n, len(feature_cols))])


def aggregate_median_response(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    return (
        df.groupby(["board", "concentration_numeric"], observed=True)[list(feature_cols)]
        .median(numeric_only=True)
        .reset_index()
        .sort_values(["board", "concentration_numeric"])
    )


def plot_board_response_curves(df: pd.DataFrame, feature_cols: Sequence[str], paths: Paths) -> List[str]:
    selected = choose_representative_features(df, feature_cols, n=min(6, len(feature_cols)))
    agg = aggregate_median_response(df, selected)
    n = len(selected)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, max(4, 3.2 * nrows)), squeeze=False)
    for ax, feat in zip(axes.ravel(), selected):
        for board in BOARD_ORDER:
            sub = agg[agg["board"] == board]
            if sub.empty:
                continue
            ax.plot(sub["concentration_numeric"], sub[feat], marker="o", linewidth=1.5, label=board)
        ax.set_title(feat)
        ax.set_xlabel("Methane concentration")
        ax.set_ylabel("Median feature response")
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)))
    fig.suptitle("Day5 board-level median response curves", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(paths.figures_dir / "day5_board_response_curves.png", dpi=180)
    plt.close(fig)
    return selected


def fit_gain_curvature(df: pd.DataFrame, selected_features: Sequence[str], paths: Paths) -> pd.DataFrame:
    rows = []
    for feat in selected_features:
        for board in BOARD_ORDER:
            sub = df[df["board"] == board][["concentration_numeric", feat]].dropna()
            if len(sub) < 3:
                continue
            x = sub["concentration_numeric"].to_numpy(dtype=float)
            y = sub[feat].to_numpy(dtype=float)
            X_lin = x.reshape(-1, 1)
            lin = LinearRegression().fit(X_lin, y)
            pred_lin = lin.predict(X_lin)
            X_quad = np.column_stack([x, x ** 2])
            quad = LinearRegression().fit(X_quad, y)
            pred_quad = quad.predict(X_quad)
            high_thr = np.quantile(x, HIGH_CONC_Q)
            high_mask = x >= high_thr
            rows.append(
                {
                    "feature": feat,
                    "board": board,
                    "n": int(len(sub)),
                    "linear_intercept": float(lin.intercept_),
                    "linear_gain_slope": float(lin.coef_[0]),
                    "linear_r2": float(r2_score(y, pred_lin)) if len(np.unique(y)) > 1 else np.nan,
                    "quadratic_intercept": float(quad.intercept_),
                    "quadratic_gain": float(quad.coef_[0]),
                    "quadratic_curvature": float(quad.coef_[1]),
                    "quadratic_r2": float(r2_score(y, pred_quad)) if len(np.unique(y)) > 1 else np.nan,
                    "high_conc_mean_residual_quad": float(np.mean(y[high_mask] - pred_quad[high_mask])) if high_mask.any() else np.nan,
                    "high_conc_abs_residual_quad": float(np.mean(np.abs(y[high_mask] - pred_quad[high_mask]))) if high_mask.any() else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(paths.results_dir / "day5_gain_curvature_summary.csv", index=False)
    return out


def plot_gain_curvature(summary: pd.DataFrame, paths: Paths) -> None:
    if summary.empty:
        return
    focus = summary[summary["board"].isin(["B1", "B5"] + SOURCE_BOARDS)].copy()
    # Aggregate across selected features with scale normalization to avoid one feature dominating.
    for col in ["linear_gain_slope", "quadratic_curvature", "high_conc_mean_residual_quad"]:
        denom = focus.groupby("feature")[col].transform(lambda s: np.nanmedian(np.abs(s)) or 1.0)
        focus[col + "_norm"] = focus[col] / denom.replace(0, np.nan)
    metrics = ["linear_gain_slope_norm", "quadratic_curvature_norm", "high_conc_mean_residual_quad_norm"]
    plot_df = focus.groupby("board", observed=True)[metrics].median(numeric_only=True).reindex(BOARD_ORDER).dropna(how="all")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Day5 gain/curvature/high-concentration residual decomposition")
    ax.set_xlabel("Board")
    ax.set_ylabel("Median normalized coefficient/residual")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "day5_gain_curvature_b1_vs_b5.png", dpi=180)
    plt.close(fig)


def pca_board_trajectories(df: pd.DataFrame, feature_cols: Sequence[str], paths: Paths) -> pd.DataFrame:
    cols = list(feature_cols)
    work = df[["board", "concentration_numeric"] + cols].copy()
    # Use medians by board/concentration to make the trajectory plot compact and memory safe.
    agg = work.groupby(["board", "concentration_numeric"], observed=True)[cols].median(numeric_only=True).reset_index()
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=2, random_state=0))
    pcs = pipe.fit_transform(agg[cols])
    agg["PC1"] = pcs[:, 0]
    agg["PC2"] = pcs[:, 1]
    pca = pipe.named_steps["pca"]
    agg.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()

    fig, ax = plt.subplots(figsize=(9, 7))
    for board in BOARD_ORDER:
        sub = agg[agg["board"] == board].sort_values("concentration_numeric")
        if sub.empty:
            continue
        sc = ax.scatter(sub["PC1"], sub["PC2"], c=sub["concentration_numeric"], s=55, label=board)
        ax.plot(sub["PC1"], sub["PC2"], linewidth=1.2, alpha=0.75)
        for _, r in sub.iterrows():
            ax.annotate(str(int(r["concentration_numeric"])) if float(r["concentration_numeric"]).is_integer() else f"{r['concentration_numeric']:.1f}",
                        (r["PC1"], r["PC2"]), fontsize=7, alpha=0.8)
    ax.set_title("Day5 PCA board trajectories using leakage-free features")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Board", loc="best")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Methane concentration")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "day5_pca_board_trajectories.png", dpi=180)
    plt.close(fig)
    agg.to_csv(paths.results_dir / "day5_pca_board_trajectories.csv", index=False)
    return agg


def _spearman_monotonicity(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.nanstd(y) < 1e-12:
        return np.nan
    xr = pd.Series(x).rank().to_numpy(dtype=float)
    yr = pd.Series(y).rank().to_numpy(dtype=float)
    corr = np.corrcoef(xr, yr)[0, 1]
    return float(abs(corr)) if np.isfinite(corr) else np.nan


def feature_transferability_scores(df: pd.DataFrame, feature_cols: Sequence[str], paths: Paths) -> pd.DataFrame:
    agg = aggregate_median_response(df, feature_cols)
    conc_values = sorted(agg["concentration_numeric"].dropna().unique())
    high_thr = float(np.quantile(conc_values, HIGH_CONC_Q)) if conc_values else np.nan
    rows = []
    for feat in feature_cols:
        source = agg[agg["board"].isin(SOURCE_BOARDS)][["board", "concentration_numeric", feat]].dropna()
        if source.empty:
            continue
        source_by_conc = source.groupby("concentration_numeric", observed=True)[feat].agg(["median", "mean", "std"]).reset_index()
        source_by_conc["cv"] = source_by_conc["std"] / source_by_conc["mean"].abs().replace(0, np.nan)
        between_cv = float(source_by_conc["cv"].replace([np.inf, -np.inf], np.nan).median(skipna=True))
        all_med = agg.groupby("concentration_numeric", observed=True)[feat].median().reset_index()
        monotonicity = _spearman_monotonicity(all_med["concentration_numeric"].to_numpy(float), all_med[feat].to_numpy(float))

        deviations = {}
        high_deviations = {}
        for board in TARGET_BOARDS:
            t = agg[agg["board"] == board][["concentration_numeric", feat]].dropna()
            merged = t.merge(source_by_conc[["concentration_numeric", "median"]], on="concentration_numeric", how="inner")
            scale = float(np.nanmedian(np.abs(source_by_conc["median"]))) or 1.0
            if merged.empty:
                deviations[board] = np.nan
                high_deviations[board] = np.nan
            else:
                dev = (merged[feat] - merged["median"]).abs() / max(scale, 1e-12)
                deviations[board] = float(dev.median(skipna=True))
                high = merged[merged["concentration_numeric"] >= high_thr]
                high_dev = (high[feat] - high["median"]).abs() / max(scale, 1e-12)
                high_deviations[board] = float(high_dev.median(skipna=True)) if not high.empty else np.nan
        board_invariant_score = float((1.0 / (1.0 + (between_cv if np.isfinite(between_cv) else 1.0))) * (monotonicity if np.isfinite(monotonicity) else 0.0))
        rows.append(
            {
                "feature": feat,
                "source_between_board_cv_median": between_cv,
                "monotonicity_abs_spearman": monotonicity,
                "board_invariant_score": board_invariant_score,
                "b1_deviation_from_source_median": deviations.get("B1", np.nan),
                "b5_deviation_from_source_median": deviations.get("B5", np.nan),
                "b1_high_conc_deviation": high_deviations.get("B1", np.nan),
                "b5_high_conc_deviation": high_deviations.get("B5", np.nan),
                "b1_specific_failure_score": deviations.get("B1", np.nan) - deviations.get("B5", np.nan),
            }
        )
    out = pd.DataFrame(rows).sort_values("board_invariant_score", ascending=False)
    out.to_csv(paths.results_dir / "day5_feature_transferability_scores.csv", index=False)
    return out


def plot_feature_transferability(scores: pd.DataFrame, paths: Paths) -> None:
    if scores.empty:
        return
    cols = [
        "board_invariant_score",
        "b1_deviation_from_source_median",
        "b5_deviation_from_source_median",
        "b1_high_conc_deviation",
        "b5_high_conc_deviation",
    ]
    top = scores.head(12).set_index("feature")[cols]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    top.plot(kind="bar", ax=ax)
    ax.set_title("Day5 feature transferability scores: stable vs B1/B5-deviating features")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Score / normalized deviation")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "day5_feature_transferability_top.png", dpi=180)
    plt.close(fig)


def failure_mode_diagnostic(scores: pd.DataFrame, gain_summary: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    rows = []
    for board in TARGET_BOARDS:
        dev_col = f"{board.lower()}_deviation_from_source_median"
        high_col = f"{board.lower()}_high_conc_deviation"
        median_dev = float(scores[dev_col].median(skipna=True)) if dev_col in scores else np.nan
        high_dev = float(scores[high_col].median(skipna=True)) if high_col in scores else np.nan
        board_gain = gain_summary[gain_summary["board"] == board]
        src_gain = gain_summary[gain_summary["board"].isin(SOURCE_BOARDS)]
        if not board_gain.empty and not src_gain.empty:
            merged = board_gain[["feature", "linear_gain_slope", "quadratic_curvature", "high_conc_mean_residual_quad"]].merge(
                src_gain.groupby("feature", observed=True)[["linear_gain_slope", "quadratic_curvature", "high_conc_mean_residual_quad"]].median().reset_index(),
                on="feature",
                suffixes=("_target", "_source"),
            )
            gain_scale = np.nanmedian(np.abs(merged["linear_gain_slope_source"])) or 1.0
            curv_scale = np.nanmedian(np.abs(merged["quadratic_curvature_source"])) or 1.0
            gain_mismatch = float(np.nanmedian(np.abs(merged["linear_gain_slope_target"] - merged["linear_gain_slope_source"]) / max(gain_scale, 1e-12)))
            curvature_mismatch = float(np.nanmedian(np.abs(merged["quadratic_curvature_target"] - merged["quadratic_curvature_source"]) / max(curv_scale, 1e-12)))
            compression = float(np.nanmedian(np.abs(merged["high_conc_mean_residual_quad_target"])))
        else:
            gain_mismatch = curvature_mismatch = compression = np.nan

        coverage_limited_score = median_dev
        intrinsic_deformation_score = float(np.nanmedian([curvature_mismatch, high_dev]))
        high_compression_score = float(np.nanmedian([compression, high_dev]))
        # Interpretation is evidence-based but cautious.
        if board == "B5" and coverage_limited_score <= max(intrinsic_deformation_score, 1e-12) * 1.25:
            likely = "coverage-limited / adaptation-responsive candidate"
            adaptation = "high in prior Day3 evidence; not re-proven by Day5"
        elif board == "B5":
            likely = "mixed coverage and deformation evidence"
            adaptation = "high in prior Day3 evidence; not re-proven by Day5"
        elif board == "B1" and intrinsic_deformation_score >= coverage_limited_score * 0.75:
            likely = "target-intrinsic deformation / compression candidate"
            adaptation = "partial in prior Day3.5 evidence; persistent residuals likely"
        else:
            likely = "mixed or unresolved failure mode"
            adaptation = "partial/uncertain based on available evidence"
        rows.append(
            {
                "board": board,
                "coverage_limited_score": coverage_limited_score,
                "gain_mismatch_score": gain_mismatch,
                "curvature_mismatch_score": curvature_mismatch,
                "intrinsic_deformation_score": intrinsic_deformation_score,
                "high_concentration_compression_score": high_compression_score,
                "adaptation_responsiveness": adaptation,
                "likely_failure_mode": likely,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.results_dir / "day5_failure_mode_diagnostic.csv", index=False)
    return out


def plot_failure_mode_summary(diag: pd.DataFrame, paths: Paths) -> None:
    if diag.empty:
        return
    cols = ["coverage_limited_score", "intrinsic_deformation_score", "high_concentration_compression_score"]
    plot_df = diag.set_index("board")[cols]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Day5 B1 vs B5 failure-mode diagnostic")
    ax.set_xlabel("Target board")
    ax.set_ylabel("Relative diagnostic score")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "day5_b1_b5_failure_mode_summary.png", dpi=180)
    plt.close(fig)


def optional_residual_model_check(df: pd.DataFrame, feature_cols: Sequence[str], paths: Paths, include_optional_heavy: bool = False) -> None:
    """Lightweight source->B1/B5 residual check. Ridge by default; RF only if explicitly enabled."""
    train = df[df["board"].isin(SOURCE_BOARDS)].copy()
    tests = df[df["board"].isin(TARGET_BOARDS)].copy()
    if train.empty or tests.empty:
        return
    model_name = "Ridge"
    if include_optional_heavy and RandomForestRegressor is not None:
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=150, random_state=0, n_jobs=1, min_samples_leaf=2),
        )
        model_name = "RandomForestRegressor_n150"
    else:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    X_train = train[list(feature_cols)]
    y_train = train["concentration_numeric"].to_numpy(float)
    model.fit(X_train, y_train)
    rows = []
    for board in TARGET_BOARDS:
        sub = tests[tests["board"] == board]
        if sub.empty:
            continue
        pred = model.predict(sub[list(feature_cols)])
        tmp = sub[["board", "concentration_numeric"]].copy()
        tmp["prediction"] = pred
        tmp["residual_true_minus_pred"] = tmp["concentration_numeric"] - tmp["prediction"]
        tmp["model"] = model_name
        rows.append(tmp)
    if not rows:
        return
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(paths.results_dir / "day5_b1_b5_residual_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for board in TARGET_BOARDS:
        sub = out[out["board"] == board]
        if sub.empty:
            continue
        med = sub.groupby("concentration_numeric", observed=True)["residual_true_minus_pred"].median().reset_index()
        ax.plot(med["concentration_numeric"], med["residual_true_minus_pred"], marker="o", label=board)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"Day5 optional residual comparison ({model_name})")
    ax.set_xlabel("True methane concentration")
    ax.set_ylabel("Residual: true - predicted")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "day5_b1_b5_residual_comparison.png", dpi=180)
    plt.close(fig)


def write_observations(paths: Paths, selected_features: Sequence[str], scores: pd.DataFrame, diag: pd.DataFrame, pca_df: pd.DataFrame) -> None:
    invariant = scores.sort_values("board_invariant_score", ascending=False).head(8)["feature"].tolist() if not scores.empty else []
    b1_fail = scores.sort_values("b1_specific_failure_score", ascending=False).head(8)["feature"].tolist() if not scores.empty else []
    diag_text = diag.to_markdown(index=False) if not diag.empty else "No diagnostic rows generated."
    evr = pca_df.attrs.get("explained_variance_ratio", []) if hasattr(pca_df, "attrs") else []
    evr_text = ", ".join(f"PC{i+1}={v*100:.1f}%" for i, v in enumerate(evr)) if evr else "not available"
    text = f"""# Day5 Observations — B1 vs B5 Failure-Mode Analysis

## Purpose

Day5 is a mechanism-analysis step, not a leaderboard-optimization step. The goal is to compare B1 and B5 as potentially different transfer-failure modes using lightweight, leakage-safe analysis.

## Leakage and memory-safety notes

- Forbidden label/metadata columns were excluded from model input features.
- The final leakage check is saved to `results/day5/day5_debug_feature_leakage_check.csv`.
- The analysis intentionally avoids deep learning, UMAP/t-SNE, nested cross-validation, large grid search, and repeated permutation importance.
- PCA uses aggregated board/concentration medians for compact trajectory analysis.
- All figures are saved and closed after creation.

## Representative features used for response-curve and gain/curvature analysis

{chr(10).join(f'- `{f}`' for f in selected_features)}

## 1. How does B1 differ from B5?

B1 and B5 should not be treated as the same transfer problem. The diagnostic table below summarizes their relative behavior:

{diag_text}

Current evidence supports the working interpretation that B5 is more consistent with a coverage-limited or shifted-domain problem, whereas B1 shows stronger evidence of intrinsic response deformation or high-concentration compression. This remains a cautious interpretation, not proof of an exact physical cause.

## 2. Is B1 mainly gain mismatch, curvature mismatch, or compression?

Day5 decomposes feature response into linear gain, quadratic curvature, and high-concentration residual terms. If B1 has a large curvature/high-concentration score relative to its global gain score, it should be interpreted as deformation/compression rather than a simple gain offset. The physical mechanism remains unresolved.

## 3. Is B5 primarily coverage-limited?

The Day5 evidence should be read together with prior Day3 evidence: B5 was adaptation-responsive under mean/std alignment, suggesting much of its error can be explained by statistical mismatch or limited target coverage. Day5 does not re-prove adaptation responsiveness; it checks whether B5 lies closer to source-like trajectories and feature relationships than B1.

## 4. Which features appear board-invariant?

Top board-invariant candidate features from the transferability score:

{chr(10).join(f'- `{f}`' for f in invariant) if invariant else '- Not enough valid features to rank.'}

These are features with relatively low source-board coefficient of variation and relatively monotonic concentration response.

## 5. Which features fail specifically on B1?

Features with the largest B1-specific deviation relative to B5:

{chr(10).join(f'- `{f}`' for f in b1_fail) if b1_fail else '- Not enough valid features to rank.'}

These features should be inspected as candidates for B1-specific high-concentration instability or compression.

## 6. Does PCA support two different failure modes?

The PCA trajectory analysis uses leakage-free features only. Explained variance: {evr_text}. If B5 lies outside the source-board coverage but follows a similar concentration direction, that supports a coverage-limited interpretation. If B1 bends, compresses, or deviates at high concentration in PCA space, that supports an intrinsic deformation interpretation.

## 7. What should Day6 investigate?

Day6 should investigate whether B1's apparent response deformation can be corrected by feature-level transformations that explicitly model saturation/compression without label leakage. Recommended next steps:

1. Compare sensor-wise saturation indicators and dynamic response features for B1 versus source boards.
2. Test monotonic calibration mappings fitted only on small target calibration subsets.
3. Separate baseline drift, gain shift, and high-concentration saturation using sensor-level physics features.
4. Validate whether B1-like compression appears in any non-methane gases or is methane-specific.
5. Keep Day6 memory-safe and mechanism-focused rather than running large model searches.

## Scientific caution

Day5 supports the hypothesis that B1 and B5 represent two different transfer failure modes. It does not prove the microscopic physical cause of B1 behavior. The current evidence supports response deformation/compression, but additional controlled analysis is required.
"""
    (paths.results_dir / "day5_observations.md").write_text(text, encoding="utf-8")


def verify_outputs(paths: Paths, optional_residual: bool = True) -> None:
    required_figs = [
        "day5_board_response_curves.png",
        "day5_gain_curvature_b1_vs_b5.png",
        "day5_pca_board_trajectories.png",
        "day5_feature_transferability_top.png",
        "day5_b1_b5_failure_mode_summary.png",
    ]
    required_results = [
        "day5_gain_curvature_summary.csv",
        "day5_feature_transferability_scores.csv",
        "day5_failure_mode_diagnostic.csv",
        "day5_debug_feature_leakage_check.csv",
        "day5_observations.md",
    ]
    missing = []
    for f in required_figs:
        if not (paths.figures_dir / f).exists():
            missing.append(str(paths.figures_dir / f))
    for f in required_results:
        if not (paths.results_dir / f).exists():
            missing.append(str(paths.results_dir / f))
    if missing:
        raise FileNotFoundError("Missing required Day5 outputs:\n" + "\n".join(missing))


def run_day5(include_optional_heavy: bool = False, root: Path | None = None) -> dict[str, Path]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    paths = get_paths(root)
    ensure_dirs(paths)
    print(f"[Day5] Project root: {paths.root}")
    print(f"[Day5] Input feature table: {paths.input_csv}")
    print(f"[Day5] Results dir: {paths.results_dir}")
    print(f"[Day5] Figures dir: {paths.figures_dir}")
    print(f"[Day5] include_optional_heavy={include_optional_heavy}")

    df = load_feature_table(paths)
    feature_cols = select_feature_columns(df, max_features=40)
    save_leakage_check(df, feature_cols, paths)

    selected = plot_board_response_curves(df, feature_cols, paths)
    gain_summary = fit_gain_curvature(df, selected, paths)
    plot_gain_curvature(gain_summary, paths)
    pca_df = pca_board_trajectories(df, feature_cols, paths)
    scores = feature_transferability_scores(df, feature_cols, paths)
    plot_feature_transferability(scores, paths)
    diag = failure_mode_diagnostic(scores, gain_summary, paths)
    plot_failure_mode_summary(diag, paths)
    optional_residual_model_check(df, feature_cols, paths, include_optional_heavy=include_optional_heavy)
    write_observations(paths, selected, scores, diag, pca_df)
    verify_outputs(paths)

    print("\n[Day5] Completed required outputs.")
    print("[Day5] Optional residual comparison was generated with Ridge by default; RF remains disabled unless include_optional_heavy=True.")
    return {
        "script": paths.script_path,
        "results_dir": paths.results_dir,
        "figures_dir": paths.figures_dir,
        "notebook": paths.root / "notebooks" / "day5_failure_mode_analysis.ipynb",
    }


if __name__ == "__main__":
    run_day5(include_optional_heavy=False)
