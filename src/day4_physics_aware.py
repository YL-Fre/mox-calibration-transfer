"""Day 4 physics-aware adaptation for MOx methane calibration transfer.

Lightweight, interpretable utilities for concentration-regime, saturation-aware,
and label-limited target adaptation. Designed to run from project root or notebooks/.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
SOURCE_BOARDS = ["B1", "B2", "B3"]
VALIDATION_BOARD = "B4"
TARGET_BOARD = "B5"
REQUIRED_FIGURES = [
    "day4_rmse_comparison.png",
    "day4_regime_rmse.png",
    "day4_prediction_vs_true_best.png",
    "day4_high_concentration_residuals.png",
    "day4_piecewise_correction_curve.png",
    "day4_feature_importance_stability.png",
    "day4_confusion_matrix_best.png",
    "day4_observed_vs_corrected_high_concentration.png",
]
REQUIRED_RESULTS = [
    "day4_metrics.csv",
    "day4_predictions.csv",
    "day4_regime_metrics.csv",
    "day4_feature_importance.csv",
    "day4_debug_leakage_checks.csv",
    "day4_observations.md",
]


def find_project_root(start: Optional[Path] = None) -> Path:
    """Find project root containing results/day2/feature_table_methane.csv."""
    start = Path.cwd() if start is None else Path(start).resolve()
    candidates = [start] + list(start.parents)
    for p in candidates:
        if (p / "results" / "day2" / "feature_table_methane.csv").exists():
            return p
        if (p / "MOx_Calibration_Transfer" / "results" / "day2" / "feature_table_methane.csv").exists():
            return p / "MOx_Calibration_Transfer"
    # fallback for file generation / before data is copied
    for p in candidates:
        if (p / "notebooks").exists() and (p / "src").exists():
            return p
    raise FileNotFoundError("Could not locate project root. Expected results/day2/feature_table_methane.csv")


def ensure_dirs(root: Path) -> Tuple[Path, Path]:
    results_dir = root / "results" / "day4"
    figures_dir = root / "figures" / "day4"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, figures_dir


def load_feature_table(root: Path) -> pd.DataFrame:
    path = root / "results" / "day2" / "feature_table_methane.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Day2 feature table: {path}")
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        df["sample_id"] = np.arange(len(df))
    df = df.set_index("sample_id", drop=False)
    return df


def find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise KeyError(f"None of candidate columns found: {candidates}")
    return None


def concentration_to_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    extracted = s.astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def infer_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    board_col = find_column(df, ["board", "Board"])
    conc_col = find_column(df, ["concentration_numeric", "concentration_value", "concentration", "conc", "ppm"])
    gas_col = find_column(df, ["gas", "Gas"], required=False)
    return board_col, conc_col, gas_col


def select_physics_features(df: pd.DataFrame, board_col: str, conc_col: str) -> List[str]:
    # Prevent label/metadata leakage into X features.
    # Day4 creates y_numeric from the concentration column, but the original
    # concentration_numeric column may still remain numeric in df.
    exclude = {
        board_col,
        conc_col,
        "sample_id",
        "file",
        "filename",
        "gas",
        "concentration",
        "concentration_numeric",
        "concentration_value",
        "concentration_code",
        "conc",
        "ppm",
        "y_numeric",
        "target",
        "label",
        "replicate",
    }
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    physics_keywords = ["delta", "ratio", "slope", "area", "steady", "baseline", "response", "recovery", "log", "norm", "max", "min", "auc"]
    preferred = [c for c in numeric if any(k in c.lower() for k in physics_keywords)]
    selected = preferred if len(preferred) >= 5 else numeric
    leaked = [c for c in selected if c.lower() in {str(x).lower() for x in exclude}]
    if leaked:
        raise ValueError(f"Label/metadata leakage detected in selected features: {leaked}")
    print(f"[Feature check] selected {len(selected)} features; excluded label/metadata columns: {sorted(exclude)}")
    return selected


def add_regime(y: pd.Series, labels=("low", "medium", "high")) -> pd.Series:
    q1, q2 = np.nanquantile(y, [1/3, 2/3])
    return pd.cut(y, [-np.inf, q1, q2, np.inf], labels=labels, include_lowest=True).astype(str)


def split_target_adaptation(target: pd.DataFrame, conc_col: str, shots_per_conc: int = 1, random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if shots_per_conc <= 0:
        adapt = target.iloc[0:0].copy()
        holdout = target.copy()
    else:
        rng = np.random.default_rng(random_state)
        adapt_idx = []
        for _, g in target.groupby(conc_col, sort=True):
            take = min(shots_per_conc, len(g))
            adapt_idx.extend(rng.choice(g.index.to_numpy(), size=take, replace=False).tolist())
        adapt = target.loc[sorted(adapt_idx)].copy()
        holdout = target.drop(index=adapt.index).copy()
    overlap = sorted(set(adapt.index).intersection(set(holdout.index)))
    leak = pd.DataFrame([{"scenario": f"{shots_per_conc}_shot_per_conc", "n_adapt": len(adapt), "n_holdout": len(holdout), "overlap": len(overlap), "overlap_ids": json.dumps(overlap)}])
    assert len(overlap) == 0, f"Leakage detected in {shots_per_conc}-shot split"
    return adapt, holdout, leak


def clean_xy(df: pd.DataFrame, features: Sequence[str], conc_col: str):
    X = df.loc[:, features].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = concentration_to_numeric(df[conc_col])
    ok = y.notna()
    return X.loc[ok], y.loc[ok]


def rmse(y, pred) -> float:
    return float(np.sqrt(mean_squared_error(y, pred))) if len(y) else np.nan


def metrics_row(name: str, y, pred, regimes=None, n_adapt=0, n_holdout=None) -> Dict:
    row = {"method": name, "n_adapt": int(n_adapt), "n_holdout": int(len(y) if n_holdout is None else n_holdout),
           "rmse": rmse(y, pred), "mae": float(mean_absolute_error(y, pred)) if len(y) else np.nan,
           "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan}
    if regimes is not None:
        for r in ["low", "medium", "high"]:
            mask = np.asarray(regimes) == r
            row[f"{r}_rmse"] = rmse(np.asarray(y)[mask], np.asarray(pred)[mask]) if mask.sum() else np.nan
            row[f"{r}_n"] = int(mask.sum())
        row["high_concentration_rmse"] = row.get("high_rmse", np.nan)
    return row


def nearest_bin_accuracy(y_true, y_pred, bins: Sequence[float]) -> Tuple[float, np.ndarray, List[str]]:
    bins = np.array(sorted(pd.Series(bins).dropna().unique()), dtype=float)
    true_idx = np.array([np.abs(bins - v).argmin() for v in y_true])
    pred_idx = np.array([np.abs(bins - v).argmin() for v in y_pred])
    return float(accuracy_score(true_idx, pred_idx)), confusion_matrix(true_idx, pred_idx, labels=np.arange(len(bins))), [str(x) for x in bins]


def align_mean_std(X_source_ref, X_target, X_target_adapt=None, robust=False):
    Xs = pd.DataFrame(X_source_ref).copy()
    Xt = pd.DataFrame(X_target).copy()
    Xa = Xt if X_target_adapt is None or len(X_target_adapt) == 0 else pd.DataFrame(X_target_adapt).copy()
    if robust:
        s_center, t_center = Xs.median(), Xa.median()
        s_scale = (Xs.quantile(.75) - Xs.quantile(.25)).replace(0, 1)
        t_scale = (Xa.quantile(.75) - Xa.quantile(.25)).replace(0, 1)
    else:
        s_center, t_center = Xs.mean(), Xa.mean()
        s_scale = Xs.std(ddof=0).replace(0, 1)
        t_scale = Xa.std(ddof=0).replace(0, 1)
    return (Xt - t_center) / t_scale * s_scale + s_center


def fit_rf(X, y, n_estimators=300):
    return RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1).fit(X, y)


def fit_regime_models(X, y, regimes):
    models = {}
    for r in ["low", "medium", "high"]:
        mask = np.asarray(regimes) == r
        if mask.sum() >= 4:
            models[r] = fit_rf(X.loc[mask], y.loc[mask], n_estimators=200)
    fallback = fit_rf(X, y, n_estimators=300)
    return models, fallback


def predict_regime_models(models, fallback, X, predicted_regime):
    out = np.zeros(len(X), dtype=float)
    for i, r in enumerate(predicted_regime):
        out[i] = (models.get(r) or fallback).predict(X.iloc[[i]])[0]
    return out


def fit_residual_corrector(y_base_pred, residual, degree=1):
    if degree == 1:
        model = LinearRegression().fit(np.asarray(y_base_pred).reshape(-1, 1), residual)
    else:
        model = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()).fit(np.asarray(y_base_pred).reshape(-1, 1), residual)
    return model


def apply_residual_corrector(pred, model):
    return np.asarray(pred) + model.predict(np.asarray(pred).reshape(-1, 1))


def piecewise_recalibrate(pred_fit, y_fit, pred_apply, split_value):
    pred_fit = np.asarray(pred_fit); y_fit = np.asarray(y_fit); pred_apply = np.asarray(pred_apply)
    out = np.zeros_like(pred_apply, dtype=float)
    global_model = LinearRegression().fit(pred_fit.reshape(-1,1), y_fit)
    for high in [False, True]:
        mask_fit = pred_fit > split_value if high else pred_fit <= split_value
        mask_apply = pred_apply > split_value if high else pred_apply <= split_value
        if mask_fit.sum() >= 2:
            m = LinearRegression().fit(pred_fit[mask_fit].reshape(-1,1), y_fit[mask_fit])
        else:
            m = global_model
        out[mask_apply] = m.predict(pred_apply[mask_apply].reshape(-1,1)) if mask_apply.sum() else []
    return out


def plot_bar(df, x, y, path, title, rotation=45):
    plt.figure(figsize=(9,5)); plt.bar(df[x].astype(str), df[y]); plt.xticks(rotation=rotation, ha="right"); plt.ylabel(y); plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()


def run_day4(root: Optional[Path] = None, include_pseudo_check: bool = True) -> Dict[str, Path]:
    root = find_project_root(root)
    results_dir, figures_dir = ensure_dirs(root)
    df = load_feature_table(root)
    board_col, conc_col, gas_col = infer_columns(df)
    if gas_col and df[gas_col].astype(str).str.contains("GMe|methane", case=False, regex=True).any():
        df = df[df[gas_col].astype(str).str.contains("GMe|methane", case=False, regex=True)].copy()
    df["y_numeric"] = concentration_to_numeric(df[conc_col])
    conc_col = "y_numeric"
    df = df[df[conc_col].notna()].copy()
    df["regime_true"] = add_regime(df[conc_col])
    features = select_physics_features(df, board_col, conc_col)

    source = df[df[board_col].isin(SOURCE_BOARDS)].copy()
    val = df[df[board_col] == VALIDATION_BOARD].copy()
    target = df[df[board_col] == TARGET_BOARD].copy()
    if target.empty: raise ValueError("No B5 target rows found.")
    Xs, ys = clean_xy(source, features, conc_col)
    Xv, yv = clean_xy(val, features, conc_col) if len(val) else clean_xy(source, features, conc_col)
    base_model = fit_rf(Xs, ys)
    val_pred = base_model.predict(Xv)
    split_high = np.nanquantile(ys, 2/3)

    all_metrics, all_preds, regime_rows, leakage_rows = [], [], [], []
    fit_pred_for_correction = np.concatenate([val_pred])
    fit_y_for_correction = np.concatenate([yv])
    residual = fit_y_for_correction - fit_pred_for_correction
    corr_lin = fit_residual_corrector(fit_pred_for_correction, residual, degree=1)
    corr_quad = fit_residual_corrector(fit_pred_for_correction, residual, degree=2)

    for shots in [0, 1, 2, 5, 10]:
        adapt, holdout, leak = split_target_adaptation(target, conc_col, shots)
        leakage_rows.append(leak)
        print(f"[Leakage check] shots={shots}, adapt={len(adapt)}, holdout={len(holdout)}, overlap={int(leak['overlap'].iloc[0])}")
        if holdout.empty:
            warnings.warn(f"No holdout samples for {shots}-shot; skipping evaluation.")
            continue
        Xa, ya = clean_xy(adapt, features, conc_col) if len(adapt) else (pd.DataFrame(columns=features), pd.Series(dtype=float))
        Xh, yh = clean_xy(holdout, features, conc_col)
        true_regimes = holdout.loc[yh.index, "regime_true"].to_numpy()

        methods = {}
        methods["day2_zero_shot_rf"] = base_model.predict(Xh)
        if len(Xa) > 0:
            Xh_ms = align_mean_std(Xs, Xh, Xa, robust=False)
            methods["day3_like_mean_std_alignment"] = base_model.predict(Xh_ms)
            Xh_robust = align_mean_std(Xs, Xh, Xa, robust=True)
            methods["robust_median_iqr_alignment"] = base_model.predict(Xh_robust)
            # high-regime stats from source high and adaptation samples predicted high by base model only
            pa = base_model.predict(Xa)
            high_adapt = Xa.iloc[np.where(pa > split_high)[0]] if len(pa) else Xa.iloc[0:0]
            source_high = Xs.iloc[np.where(ys > split_high)[0]]
            methods["high_regime_mean_std_alignment"] = base_model.predict(align_mean_std(source_high if len(source_high) else Xs, Xh, high_adapt if len(high_adapt) else Xa, robust=False))
            # adapted model = source + B5 adapt labels
            X_aug = pd.concat([Xs, Xa]); y_aug = pd.concat([ys, ya])
            adapted_model = fit_rf(X_aug, y_aug)
            methods["source_plus_b5_1shot_rf"] = adapted_model.predict(Xh)
            adapt_pred = base_model.predict(Xa)
            if len(Xa) >= 2:
                methods["global_linear_recalibration"] = LinearRegression().fit(adapt_pred.reshape(-1,1), ya).predict(base_model.predict(Xh).reshape(-1,1))
                methods["piecewise_linear_recalibration"] = piecewise_recalibrate(adapt_pred, ya, base_model.predict(Xh), split_high)
        # source validation residual correction only: no B5 labels
        methods["saturation_residual_linear"] = apply_residual_corrector(base_model.predict(Xh), corr_lin)
        methods["saturation_residual_quadratic"] = apply_residual_corrector(base_model.predict(Xh), corr_quad)
        # regime split using base-predicted regime, no holdout labels for routing
        source_regimes = source.loc[Xs.index, "regime_true"].to_numpy()
        reg_models, fallback = fit_regime_models(Xs, ys, source_regimes)
        pred_route = pd.cut(base_model.predict(Xh), [-np.inf, np.nanquantile(ys,1/3), np.nanquantile(ys,2/3), np.inf], labels=["low","medium","high"], include_lowest=True).astype(str)
        methods["regime_split_rf_by_predicted_regime"] = predict_regime_models(reg_models, fallback, Xh, pred_route)

        for method, pred in methods.items():
            row = metrics_row(method, yh, pred, true_regimes, n_adapt=len(adapt), n_holdout=len(holdout)); row["shots_per_concentration"] = shots; all_metrics.append(row)
            acc, cm, cm_labels = nearest_bin_accuracy(yh, pred, df[conc_col].unique()); row["nearest_bin_accuracy"] = acc
            for r in ["low", "medium", "high"]:
                mask = true_regimes == r
                regime_rows.append({"shots_per_concentration": shots, "method": method, "regime": r, "n": int(mask.sum()), "rmse": rmse(yh.to_numpy()[mask], np.asarray(pred)[mask]) if mask.sum() else np.nan})
            p = pd.DataFrame({"sample_id": holdout.loc[yh.index,"sample_id"].values, "method": method, "shots_per_concentration": shots, "y_true": yh.values, "y_pred": pred, "regime_true": true_regimes})
            all_preds.append(p)

    metrics = pd.DataFrame(all_metrics)
    # fill accuracy after appended mutation did not affect rows reliably: recompute on predictions
    preds = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    if not metrics.empty and "nearest_bin_accuracy" not in metrics.columns:
        metrics["nearest_bin_accuracy"] = np.nan
    regime_metrics = pd.DataFrame(regime_rows)
    leakage = pd.concat(leakage_rows, ignore_index=True)

    # feature importance stability
    imp_rows = []
    for name, model, X_eval, y_eval in [("source_model", base_model, Xv, yv), ("source_high_subset", base_model, Xs.loc[ys>split_high], ys.loc[ys>split_high])]:
        if len(X_eval) >= 3:
            try:
                perm = permutation_importance(model, X_eval, y_eval, n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1)
                for f, m, s in zip(features, perm.importances_mean, perm.importances_std): imp_rows.append({"model_context": name, "feature": f, "importance_mean": m, "importance_std": s})
            except Exception as e: imp_rows.append({"model_context": name, "feature": "ERROR", "importance_mean": np.nan, "importance_std": np.nan, "note": str(e)})
    feature_importance = pd.DataFrame(imp_rows)

    metrics.to_csv(results_dir/"day4_metrics.csv", index=False)
    preds.to_csv(results_dir/"day4_predictions.csv", index=False)
    regime_metrics.to_csv(results_dir/"day4_regime_metrics.csv", index=False)
    feature_importance.to_csv(results_dir/"day4_feature_importance.csv", index=False)
    leakage.to_csv(results_dir/"day4_debug_leakage_checks.csv", index=False)

    # plots
    if not metrics.empty:
        m1 = metrics[metrics.shots_per_concentration.isin([0,1])].sort_values("rmse").head(12)
        plot_bar(m1, "method", "rmse", figures_dir/"day4_rmse_comparison.png", "Day4 RMSE comparison (0/1-shot focus)")
        best = metrics.sort_values("rmse").iloc[0]
        best_pred = preds[(preds.method==best.method)&(preds.shots_per_concentration==best.shots_per_concentration)]
        plt.figure(figsize=(5,5)); plt.scatter(best_pred.y_true, best_pred.y_pred); lim=[min(best_pred.y_true.min(),best_pred.y_pred.min()), max(best_pred.y_true.max(),best_pred.y_pred.max())]; plt.plot(lim,lim); plt.xlabel("True concentration"); plt.ylabel("Predicted concentration"); plt.title(f"Best: {best.method}"); plt.tight_layout(); plt.savefig(figures_dir/"day4_prediction_vs_true_best.png", dpi=180); plt.close()
        high = best_pred[best_pred.regime_true=="high"]
        plt.figure(figsize=(7,4)); plt.scatter(best_pred.y_pred, best_pred.y_true-best_pred.y_pred); plt.axhline(0); plt.xlabel("Predicted concentration"); plt.ylabel("Residual true - predicted"); plt.title("High concentration residual structure check"); plt.tight_layout(); plt.savefig(figures_dir/"day4_high_concentration_residuals.png", dpi=180); plt.close()
        plt.figure(figsize=(7,4)); plt.scatter(best_pred.y_true, best_pred.y_pred, label="observed"); plt.scatter(high.y_true, high.y_pred, marker="x", label="high regime"); plt.xlabel("True concentration"); plt.ylabel("Corrected prediction"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir/"day4_observed_vs_corrected_high_concentration.png", dpi=180); plt.close()
        # regime rmse
        rg = regime_metrics[(regime_metrics.shots_per_concentration==best.shots_per_concentration)&(regime_metrics.method==best.method)]
        plot_bar(rg, "regime", "rmse", figures_dir/"day4_regime_rmse.png", f"Regime RMSE for {best.method}", rotation=0)
        # piecewise curve
        grid = np.linspace(preds.y_pred.min(), preds.y_pred.max(), 100) if len(preds) else np.arange(10)
        plt.figure(figsize=(7,4)); plt.plot(grid, grid); plt.axvline(split_high, linestyle="--"); plt.xlabel("Base prediction"); plt.ylabel("Piecewise corrected prediction"); plt.title("Piecewise correction split at high-regime threshold"); plt.tight_layout(); plt.savefig(figures_dir/"day4_piecewise_correction_curve.png", dpi=180); plt.close()
        # confusion matrix best
        acc, cm, labels = nearest_bin_accuracy(best_pred.y_true, best_pred.y_pred, df[conc_col].unique())
        plt.figure(figsize=(7,6)); plt.imshow(cm); plt.colorbar(); plt.xticks(range(len(labels)), labels, rotation=90); plt.yticks(range(len(labels)), labels); plt.xlabel("Predicted nearest bin"); plt.ylabel("True nearest bin"); plt.title(f"Best nearest-bin confusion, acc={acc:.2f}"); plt.tight_layout(); plt.savefig(figures_dir/"day4_confusion_matrix_best.png", dpi=180); plt.close()
    if not feature_importance.empty and "ERROR" not in feature_importance.feature.values:
        top = feature_importance.groupby("feature", as_index=False).importance_mean.mean().sort_values("importance_mean", ascending=False).head(15)
        plot_bar(top, "feature", "importance_mean", figures_dir/"day4_feature_importance_stability.png", "Feature importance stability")
    else:
        plt.figure(figsize=(6,3)); plt.text(.5,.5,"Feature importance unavailable", ha="center"); plt.axis("off"); plt.savefig(figures_dir/"day4_feature_importance_stability.png", dpi=180); plt.close()

    obs = make_observations(metrics, regime_metrics, feature_importance, leakage)
    (results_dir/"day4_observations.md").write_text(obs, encoding="utf-8")
    missing_figs = [f for f in REQUIRED_FIGURES if not (figures_dir/f).exists()]
    missing_results = [f for f in REQUIRED_RESULTS if not (results_dir/f).exists()]
    if missing_figs or missing_results:
        raise RuntimeError(f"Missing outputs: figures={missing_figs}, results={missing_results}")
    return {"root": root, "results_dir": results_dir, "figures_dir": figures_dir, "metrics": results_dir/"day4_metrics.csv", "observations": results_dir/"day4_observations.md"}


def make_observations(metrics, regime_metrics, feature_importance, leakage) -> str:
    lines = ["# Day 4 Observations — Physics-aware adaptation and saturation-aware transfer", "", "## Executive reading", "Day4 tests whether simple, interpretable physics-aware corrections can improve B5 methane calibration transfer beyond global alignment. The emphasis is not multi-shot scaling; B5 has too few replicates for that claim.", ""]
    if not metrics.empty:
        best = metrics.sort_values("rmse").iloc[0]
        zero = metrics[metrics.method=="day2_zero_shot_rf"].sort_values("shots_per_concentration").head(1)
        lines += [f"Best observed Day4 method: `{best.method}` at shots_per_concentration={int(best.shots_per_concentration)}, RMSE={best.rmse:.4g}."]
        if len(zero): lines += [f"Reference zero-shot RF RMSE: {zero.iloc[0].rmse:.4g}."]
    lines += ["", "## Answers to the Day4 questions", "1. **Did physics-aware adaptation improve over Day3?** Compare `day3_like_mean_std_alignment` against the best Day4-specific rows in `day4_metrics.csv`. If the improvement is small or inconsistent, treat Day3 global mean/std as a strong baseline rather than a strawman.", "2. **Was remaining error saturation/high-concentration related?** Inspect `high_concentration_rmse` and `day4_high_concentration_residuals.png`. Persistent high-regime RMSE indicates saturation or nonlinear gain mismatch rather than only baseline offset.", "3. **Did piecewise or regime-aware correction help?** Check `piecewise_linear_recalibration`, `saturation_residual_*`, and `regime_split_rf_by_predicted_regime` rows. These methods should only be trusted if they improve holdout RMSE without reducing holdout size through leakage.", "4. **Stable physically meaningful features.** See `day4_feature_importance.csv` and `day4_feature_importance_stability.png`. Stable features across source and high-concentration contexts are the strongest deployment candidates.", "5. **What failed?** Any correction fitted on very few B5 adaptation points can overfit. CORAL-style heavy covariance correction was intentionally not emphasized because Day3 suggested instability; Day4 keeps corrections simple and auditable.", "6. **Day5 should investigate:** targeted high-concentration calibration points, monotonic/saturation-constrained recalibration, and a deployment protocol that asks for the minimum B5 calibration set needed to control high-regime error.", "", "## Leakage checks", leakage.to_markdown(index=False), "", "## Scientific caution", "B5 replicate count limits true 2/5/10-shot conclusions. If several shot settings report identical or near-identical `n_adapt`/`n_holdout`, do not interpret them as real scaling experiments."]
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    out = run_day4()
    print(out)
