"""
Day 3: Few-shot calibration transfer / adaptation for MOx methane sensors.

Input:
    results/day2/feature_table_methane.csv
Outputs:
    results/day3/*.csv, results/day3/day3_observations.md
    figures/day3/*.png

Designed for project root:
MOx_Calibration_Transfer/
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42
TRAIN_BOARDS = ["B1", "B2", "B3"]
VAL_BOARD = "B4"
TEST_BOARD = "B5"
SHOT_SIZES = [0, 1, 2, 5, 10]


@dataclass
class ProjectPaths:
    root: Path
    feature_table: Path
    results_dir: Path
    figures_dir: Path


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find a plausible project root from cwd or parent directories."""
    start = Path.cwd() if start is None else Path(start).resolve()
    candidates = [start, *start.parents]
    for c in candidates:
        if (c / "results" / "day2" / "feature_table_methane.csv").exists():
            return c
        if c.name == "MOx_Calibration_Transfer":
            return c
    return start


def get_paths(root: Optional[Path] = None) -> ProjectPaths:
    root = find_project_root(root)
    return ProjectPaths(
        root=root,
        feature_table=root / "results" / "day2" / "feature_table_methane.csv",
        results_dir=root / "results" / "day3",
        figures_dir=root / "figures" / "day3",
    )


def ensure_dirs(paths: ProjectPaths) -> None:
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)


def load_feature_table(paths: ProjectPaths) -> pd.DataFrame:
    if not paths.feature_table.exists():
        raise FileNotFoundError(
            f"Missing Day2 feature table: {paths.feature_table}\n"
            "Run Day2 first or copy feature_table_methane.csv into results/day2/."
        )
    df = pd.read_csv(paths.feature_table)
    if "board" not in df.columns:
        raise ValueError("feature_table_methane.csv must contain a 'board' column.")
    y_col = infer_regression_target(df)
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=["board", y_col]).copy()
    df["board"] = df["board"].astype(str)
    df["concentration_label"] = infer_concentration_label(df, y_col)
    return df


def infer_regression_target(df: pd.DataFrame) -> str:
    candidates = [
        "concentration_ppm", "concentration_value", "concentration_numeric",
        "conc_ppm", "ppm", "target", "y", "concentration_code",
    ]
    for c in candidates:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0:
            return c
    if "concentration" in df.columns:
        parsed = df["concentration"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0]
        if parsed.notna().sum() > 0:
            df["concentration_numeric_from_label"] = pd.to_numeric(parsed, errors="coerce")
            return "concentration_numeric_from_label"
    raise ValueError(
        "Could not infer numeric concentration target. Expected one of: "
        "concentration_code, concentration_ppm, conc_ppm, ppm, etc."
    )


def infer_concentration_label(df: pd.DataFrame, y_col: str) -> pd.Series:
    if "concentration" in df.columns:
        return df["concentration"].astype(str)
    return df[y_col].astype(str)


def infer_feature_columns(df: pd.DataFrame, y_col: str, feature_mode: str = "physics") -> List[str]:
    exclude = {
        "file", "filename", "board", "gas", "concentration", "concentration_label",
        "concentration_code", "concentration_ppm", "concentration_value",
        "concentration_numeric", "conc_ppm", "ppm", "target", "y",
        "replicate", "run", "sample_id", y_col,
    }
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    physics_keywords = [
        "ratio", "rs_r0", "rs/r0", "delta", "rel", "norm", "baseline", "slope",
        "response", "recovery", "area", "auc", "log", "max", "min", "steady", "drift",
    ]
    physics_cols = [c for c in numeric_cols if any(k in c.lower() for k in physics_keywords)]
    if feature_mode == "physics" and len(physics_cols) >= 3:
        return physics_cols
    return numeric_cols


def clean_xy(df: pd.DataFrame, feature_cols: Sequence[str], y_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    X = X.fillna(X.median(numeric_only=True))
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    return X, y


def split_b5_adapt_holdout(
    df_b5: pd.DataFrame, shots: int, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Take up to `shots` labeled B5 examples per concentration label for adaptation.

    Important:
    - adapt samples are used for few-shot adaptation
    - holdout samples are used only for final evaluation
    - adapt and holdout must never overlap
    """

    if shots <= 0:
        adapt = df_b5.iloc[0:0].copy()
        holdout = df_b5.copy()

        overlap = set(adapt.index).intersection(set(holdout.index))
        print(
            f"[Leakage check] shots={shots}, "
            f"adapt={len(adapt)}, "
            f"holdout={len(holdout)}, "
            f"overlap={len(overlap)}"
        )
        assert len(overlap) == 0, "Data leakage detected!"

        return adapt, holdout

    rng = np.random.default_rng(random_state + shots)
    adapt_indices = []

    for _, group in df_b5.groupby("concentration_label", sort=True):
        n = min(shots, max(len(group) - 1, 0))  # always keep holdout if possible

        if n > 0:
            chosen = rng.choice(
                group.index.to_numpy(),
                size=n,
                replace=False
            ).tolist()
            adapt_indices.extend(chosen)

    adapt = df_b5.loc[adapt_indices].copy()
    holdout = df_b5.drop(index=adapt_indices).copy()

    overlap = set(adapt.index).intersection(set(holdout.index))
    print(
        f"[Leakage check] shots={shots}, "
        f"adapt={len(adapt)}, "
        f"holdout={len(holdout)}, "
        f"overlap={len(overlap)}"
    )
    assert len(overlap) == 0, "Data leakage detected!"

    return adapt, holdout


def make_base_regressor() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        )),
    ])


def make_base_classifier() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1,
            class_weight="balanced_subsample"
        )),
    ])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": safe_r2(y_true, y_pred),
    }


def nearest_label_predictions(y_pred: np.ndarray, labels: Sequence[float]) -> np.ndarray:
    labels = np.array(sorted(pd.unique(np.asarray(labels))), dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return labels[np.abs(y_pred[:, None] - labels[None, :]).argmin(axis=1)]


def linear_recalibration(y_base_adapt: np.ndarray, y_adapt: np.ndarray, y_base_holdout: np.ndarray) -> np.ndarray:
    if len(y_adapt) < 2 or len(np.unique(y_base_adapt)) < 2:
        delta = float(np.mean(y_adapt - y_base_adapt)) if len(y_adapt) else 0.0
        return y_base_holdout + delta
    lr = LinearRegression().fit(y_base_adapt.reshape(-1, 1), y_adapt)
    return lr.predict(y_base_holdout.reshape(-1, 1))


def residual_correction(X_adapt: pd.DataFrame, residual: np.ndarray, X_holdout: pd.DataFrame, alpha: float = 10.0) -> np.ndarray:
    if len(X_adapt) < 2:
        return np.zeros(len(X_holdout)) + (float(np.mean(residual)) if len(residual) else 0.0)
    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    model.fit(X_adapt, residual)
    return model.predict(X_holdout)


def mean_std_align(X: pd.DataFrame, source_ref: pd.DataFrame, target_adapt: pd.DataFrame) -> pd.DataFrame:
    if len(target_adapt) < 2:
        return X.copy()
    src_mu, src_sd = source_ref.mean(), source_ref.std().replace(0, 1)
    tgt_mu, tgt_sd = target_adapt.mean(), target_adapt.std().replace(0, 1)
    return (X - tgt_mu) / tgt_sd * src_sd + src_mu


def centroid_align(X: pd.DataFrame, source_ref: pd.DataFrame, target_adapt: pd.DataFrame) -> pd.DataFrame:
    if len(target_adapt) < 1:
        return X.copy()
    shift = source_ref.mean() - target_adapt.mean()
    return X + shift


def coral_align(X: pd.DataFrame, source_ref: pd.DataFrame, target_adapt: pd.DataFrame, eps: float = 1e-3) -> pd.DataFrame:
    """Simple target-to-source CORAL alignment. Falls back if too few target samples."""
    if len(target_adapt) < 3 or X.shape[1] < 2:
        return X.copy()
    cols = X.columns
    Xv = X.to_numpy(dtype=float)
    src = source_ref.to_numpy(dtype=float)
    tgt = target_adapt.to_numpy(dtype=float)
    src_mu, tgt_mu = src.mean(axis=0), tgt.mean(axis=0)
    Cs = np.cov(src, rowvar=False) + eps * np.eye(src.shape[1])
    Ct = np.cov(tgt, rowvar=False) + eps * np.eye(tgt.shape[1])
    def mat_power(C, p):
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, eps, None)
        return vecs @ np.diag(vals ** p) @ vecs.T
    A = mat_power(Ct, -0.5) @ mat_power(Cs, 0.5)
    aligned = (Xv - tgt_mu) @ A + src_mu
    return pd.DataFrame(aligned, index=X.index, columns=cols)


def train_weighted_rf(X_train, y_train, X_adapt, y_adapt, target_weight: float = 8.0) -> Pipeline:
    X_all = pd.concat([X_train, X_adapt], axis=0)
    y_all = np.concatenate([y_train, y_adapt])
    sample_weight = np.concatenate([np.ones(len(y_train)), np.full(len(y_adapt), target_weight)])
    model = make_base_regressor()
    model.fit(X_all, y_all, rf__sample_weight=sample_weight)
    return model


def run_day3(root: Optional[str | Path] = None, feature_mode: str = "physics") -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = get_paths(Path(root) if root else None)
    ensure_dirs(paths)
    df = load_feature_table(paths)
    y_col = infer_regression_target(df)
    feature_cols = infer_feature_columns(df, y_col, feature_mode=feature_mode)
    if len(feature_cols) < 2:
        raise ValueError(f"Too few usable numeric feature columns found: {feature_cols}")

    source_df = df[df["board"].isin(TRAIN_BOARDS)].copy()
    val_df = df[df["board"].eq(VAL_BOARD)].copy()
    b5_df = df[df["board"].eq(TEST_BOARD)].copy()
    if source_df.empty or b5_df.empty:
        raise ValueError("Expected source boards B1/B2/B3 and target board B5 in feature table.")

    X_source, y_source = clean_xy(source_df, feature_cols, y_col)
    X_b5_all, y_b5_all = clean_xy(b5_df, feature_cols, y_col)
    all_labels = sorted(pd.unique(df[y_col].dropna()))

    base_model = make_base_regressor().fit(X_source, y_source)
    le = LabelEncoder().fit(df["concentration_label"].astype(str))
    clf = make_base_classifier().fit(X_source, le.transform(source_df["concentration_label"].astype(str)))

    records = []
    pred_frames = []

    for shots in SHOT_SIZES:
        adapt_df, holdout_df = split_b5_adapt_holdout(b5_df, shots)

        # Safety check: few-shot B5 adaptation samples must never appear in B5 holdout.
        # This guards against target-label leakage during final B5 evaluation.
        adaptation_idx = set(adapt_df.index)
        holdout_idx = set(holdout_df.index)
        overlap_idx = adaptation_idx.intersection(holdout_idx)
        if overlap_idx:
            raise RuntimeError(
                f"Data leakage detected for {shots}-shot adaptation: "
                f"{len(overlap_idx)} B5 samples appear in both adaptation and holdout."
            )

        X_adapt, y_adapt = clean_xy(adapt_df, feature_cols, y_col)
        X_holdout, y_holdout = clean_xy(holdout_df, feature_cols, y_col)
        if len(holdout_df) == 0:
            continue

        base_hold = base_model.predict(X_holdout)
        base_adapt = base_model.predict(X_adapt) if len(X_adapt) else np.array([])

        method_preds: Dict[str, np.ndarray] = {"no_adaptation": base_hold}
        method_preds["linear_recalibration"] = linear_recalibration(base_adapt, y_adapt, base_hold)
        method_preds["residual_ridge"] = base_hold + residual_correction(X_adapt, y_adapt - base_adapt, X_holdout)

        X_hold_centroid = centroid_align(X_holdout, X_source, X_adapt)
        method_preds["centroid_alignment"] = base_model.predict(X_hold_centroid)

        X_hold_ms = mean_std_align(X_holdout, X_source, X_adapt)
        method_preds["mean_std_alignment"] = base_model.predict(X_hold_ms)

        X_hold_coral = coral_align(X_holdout, X_source, X_adapt)
        method_preds["coral_alignment"] = base_model.predict(X_hold_coral)

        if len(X_adapt) > 0:
            rf_ft = make_base_regressor().fit(pd.concat([X_source, X_adapt]), np.concatenate([y_source, y_adapt]))
            method_preds["rf_retraining"] = rf_ft.predict(X_holdout)
            rf_w = train_weighted_rf(X_source, y_source, X_adapt, y_adapt)
            method_preds["rf_weighted_retraining"] = rf_w.predict(X_holdout)

        for method, pred in method_preds.items():
            reg = evaluate_regression(y_holdout, pred)
            cls_pred_numeric = nearest_label_predictions(pred, all_labels)
            cls_true_numeric = nearest_label_predictions(y_holdout, all_labels)
            acc = accuracy_score(cls_true_numeric, cls_pred_numeric)
            high_cut = np.percentile(pd.to_numeric(df[y_col], errors="coerce").dropna(), 75)
            high_mask = y_holdout >= high_cut
            high_rmse = rmse(y_holdout[high_mask], pred[high_mask]) if high_mask.sum() else np.nan
            records.append({
                "shots_per_concentration": shots,
                "n_adapt": len(adapt_df),
                "n_holdout": len(holdout_df),
                "method": method,
                **reg,
                "accuracy_from_regression_bins": float(acc),
                "high_concentration_rmse": high_rmse,
                "feature_mode": feature_mode,
                "n_features": len(feature_cols),
            })
            pf = holdout_df[["board", "concentration_label"]].copy()
            pf["shots_per_concentration"] = shots
            pf["method"] = method
            pf["y_true"] = y_holdout
            pf["y_pred"] = pred
            pf["residual"] = y_holdout - pred
            pred_frames.append(pf)

    metrics = pd.DataFrame(records).sort_values(["shots_per_concentration", "rmse", "method"])
    preds = pd.concat(pred_frames, ignore_index=True)
    metrics.to_csv(paths.results_dir / "day3_adaptation_metrics.csv", index=False)
    preds.to_csv(paths.results_dir / "day3_predictions.csv", index=False)
    pd.Series(feature_cols, name="feature").to_csv(paths.results_dir / "day3_feature_columns.csv", index=False)

    make_plots(paths, df, source_df, b5_df, X_source, y_source, feature_cols, y_col, metrics, preds, base_model)
    write_observations(paths, metrics, feature_cols, y_col)
    return metrics, preds


def make_plots(paths, df, source_df, b5_df, X_source, y_source, feature_cols, y_col, metrics, preds, base_model):
    # RMSE vs shots
    plt.figure(figsize=(9, 5.5))
    for method, g in metrics.groupby("method"):
        g = g.sort_values("shots_per_concentration")
        plt.plot(g["shots_per_concentration"], g["rmse"], marker="o", label=method)
    plt.xlabel("B5 adaptation samples per concentration")
    plt.ylabel("B5 holdout RMSE")
    plt.title("Few-shot adaptation efficiency on B5")
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "rmse_vs_fewshot_samples.png", dpi=200)
    plt.close()

    # Best adapted prediction vs true
    best = metrics[metrics["method"] != "no_adaptation"].sort_values("rmse").head(1)
    if not best.empty:
        row = best.iloc[0]
        p = preds[(preds["method"] == row["method"]) & (preds["shots_per_concentration"] == row["shots_per_concentration"])]
        plt.figure(figsize=(6, 6))
        plt.scatter(p["y_true"], p["y_pred"], alpha=0.75)
        lo, hi = min(p["y_true"].min(), p["y_pred"].min()), max(p["y_true"].max(), p["y_pred"].max())
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("True concentration")
        plt.ylabel("Predicted concentration")
        plt.title(f"Prediction vs true: {row['method']}, {int(row['shots_per_concentration'])}-shot")
        plt.tight_layout()
        plt.savefig(paths.figures_dir / "prediction_vs_true_best_adaptation.png", dpi=200)
        plt.close()

    # Residual reduction
    base = metrics[metrics["method"] == "no_adaptation"][["shots_per_concentration", "rmse"]].rename(columns={"rmse": "baseline_rmse"})
    imp = metrics.merge(base, on="shots_per_concentration", how="left")
    imp["rmse_reduction"] = imp["baseline_rmse"] - imp["rmse"]
    plt.figure(figsize=(9, 5.5))
    for method, g in imp[imp["method"] != "no_adaptation"].groupby("method"):
        g = g.sort_values("shots_per_concentration")
        plt.plot(g["shots_per_concentration"], g["rmse_reduction"], marker="o", label=method)
    plt.axhline(0, linestyle="--")
    plt.xlabel("B5 adaptation samples per concentration")
    plt.ylabel("RMSE reduction vs no adaptation")
    plt.title("Residual error reduction from few-shot adaptation")
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "residual_error_reduction.png", dpi=200)
    plt.close()

    # High concentration
    plt.figure(figsize=(9, 5.5))
    for method, g in metrics.groupby("method"):
        g = g.sort_values("shots_per_concentration")
        plt.plot(g["shots_per_concentration"], g["high_concentration_rmse"], marker="o", label=method)
    plt.xlabel("B5 adaptation samples per concentration")
    plt.ylabel("High concentration RMSE")
    plt.title("High-concentration transfer after adaptation")
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "high_concentration_improvement.png", dpi=200)
    plt.close()

    # Feature distribution alignment for most variable feature
    b5_X, _ = clean_xy(b5_df, feature_cols, y_col)
    variances = X_source.var().sort_values(ascending=False)
    f = variances.index[0]
    adapt_df, _ = split_b5_adapt_holdout(b5_df, 5)
    X_adapt, _ = clean_xy(adapt_df, feature_cols, y_col)
    aligned_b5 = mean_std_align(b5_X, X_source, X_adapt)
    plt.figure(figsize=(8, 5))
    plt.hist(X_source[f], bins=30, alpha=0.45, label="source B1-B3")
    plt.hist(b5_X[f], bins=30, alpha=0.45, label="B5 before")
    plt.hist(aligned_b5[f], bins=30, alpha=0.45, label="B5 after mean/std")
    plt.xlabel(f)
    plt.ylabel("count")
    plt.title("Feature distribution alignment example")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "feature_distribution_alignment.png", dpi=200)
    plt.close()

    # PCA before/after simple alignment
    combined = pd.concat([source_df, b5_df], axis=0)
    X_comb, _ = clean_xy(combined, feature_cols, y_col)
    scaler = StandardScaler().fit(X_comb)
    Z = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaler.transform(X_comb))
    plot_pca_df = combined[["board", "concentration_label"]].copy()
    plot_pca_df["PC1"] = Z[:, 0]
    plot_pca_df["PC2"] = Z[:, 1]
    plt.figure(figsize=(7.5, 6))
    for board, g in plot_pca_df.groupby("board"):
        plt.scatter(g["PC1"], g["PC2"], s=24, alpha=0.75, label=board)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA before B5 adaptation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "pca_before_adaptation.png", dpi=200)
    plt.close()

    X_b5_aligned = mean_std_align(b5_X, X_source, X_adapt)
    X_after = pd.concat([X_source, X_b5_aligned], axis=0)
    meta_after = pd.concat([source_df[["board", "concentration_label"]], b5_df[["board", "concentration_label"]]], axis=0)
    Z2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(StandardScaler().fit_transform(X_after))
    meta_after = meta_after.copy(); meta_after["PC1"] = Z2[:, 0]; meta_after["PC2"] = Z2[:, 1]
    plt.figure(figsize=(7.5, 6))
    for board, g in meta_after.groupby("board"):
        plt.scatter(g["PC1"], g["PC2"], s=24, alpha=0.75, label=board)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA after simple B5 mean/std alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.figures_dir / "pca_after_mean_std_alignment.png", dpi=200)
    plt.close()

    # Confusion matrix for best method
    if not best.empty:
        row = best.iloc[0]
        p = preds[(preds["method"] == row["method"]) & (preds["shots_per_concentration"] == row["shots_per_concentration"])]
        labels = np.array(sorted(pd.unique(pd.to_numeric(df[y_col], errors="coerce").dropna())))
        yt = nearest_label_predictions(p["y_true"].to_numpy(), labels)
        yp = nearest_label_predictions(p["y_pred"].to_numpy(), labels)
        cm = confusion_matrix(yt, yp, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(paths.results_dir / "day3_confusion_matrix_best.csv")
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, aspect="auto")
        plt.colorbar(label="count")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted concentration bin")
        plt.ylabel("True concentration bin")
        plt.title("Confusion matrix from regression bins")
        plt.tight_layout()
        plt.savefig(paths.figures_dir / "confusion_matrix_best_adaptation.png", dpi=200)
        plt.close()


def write_observations(paths: ProjectPaths, metrics: pd.DataFrame, feature_cols: Sequence[str], y_col: str) -> None:
    best_by_shot = metrics.sort_values("rmse").groupby("shots_per_concentration").head(1)
    overall_best = metrics[metrics["method"] != "no_adaptation"].sort_values("rmse").head(1)
    baseline0 = metrics[(metrics["shots_per_concentration"] == 0) & (metrics["method"] == "no_adaptation")]
    lines = []
    lines.append("# Day 3 Observations: Few-shot B5 Adaptation\n")
    lines.append(f"- Target variable inferred: `{y_col}`.")
    lines.append(f"- Primary feature mode: physics-informed / numeric selected features, n = {len(feature_cols)}.")
    lines.append("- Source training boards: B1/B2/B3; target transfer board: B5; B5 holdout labels are never used for fitting.")
    if not baseline0.empty:
        lines.append(f"- 0-shot baseline RMSE: {baseline0.iloc[0]['rmse']:.4g}; MAE: {baseline0.iloc[0]['mae']:.4g}; R²: {baseline0.iloc[0]['r2']:.4g}.")
    lines.append("\n## Best method by few-shot size\n")
    for _, r in best_by_shot.sort_values("shots_per_concentration").iterrows():
        lines.append(f"- {int(r['shots_per_concentration'])}-shot: `{r['method']}` | RMSE={r['rmse']:.4g}, MAE={r['mae']:.4g}, R²={r['r2']:.4g}, high-conc RMSE={r['high_concentration_rmse']:.4g}.")
    if not overall_best.empty:
        r = overall_best.iloc[0]
        lines.append("\n## Concise scientific summary\n")
        lines.append(f"- Most effective observed adaptation: `{r['method']}` at {int(r['shots_per_concentration'])}-shot per concentration.")
        lines.append("- Linear/recalibration-style methods test whether B5 mainly has offset/gain mismatch; feature alignment methods test baseline and distribution shift; RF retraining tests whether a few B5 labels can locally correct nonlinear board deformation.")
        lines.append("- If high-concentration RMSE remains high after adaptation, the likely failure mode is not just baseline shift but concentration-dependent saturation or nonlinear response compression.")
        lines.append("- Practical implication: a small calibration set on the deployment board can materially improve transfer, but robust high-concentration deployment still needs targeted calibration points and possibly explicit saturation-aware features.")
        lines.append("- Future work: repeat few-shot sampling over multiple random seeds, compare physics-informed vs raw/transient feature sets, and design an active calibration strategy that selects the most informative B5 concentrations.")
    (paths.results_dir / "day3_observations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="Project root. Defaults to auto-detection from cwd.")
    parser.add_argument("--feature-mode", type=str, default="physics", choices=["physics", "all"])
    args = parser.parse_args()
    metrics, _ = run_day3(root=args.root, feature_mode=args.feature_mode)
    print("Day3 complete.")
    print(metrics.sort_values("rmse").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
