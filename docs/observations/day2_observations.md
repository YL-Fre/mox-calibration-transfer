# Day 2 Observations — Methane Calibration Transfer Baseline

## Scope

Day 2 built a methane-only baseline for UCI #361 Twin Gas Sensor Arrays using simple, interpretable models only.

## Split

- Train boards: B1, B2, B3
- Validation board: B4
- Test board: B5

## Feature sets

- Raw features: 80
- Physics-informed features: 88
- Combined features: 168

## Best B5 regression result

- Model: XGBRegressor
- Feature set: physics
- MAE: 3.6183
- RMSE: 5.9375
- R²: 0.9573

## Raw vs physics-informed comparison on B5

Best raw-feature model:

- Model: XGBRegressor
- RMSE: 8.0918
- R²: 0.9206

Best physics-informed model:

- Model: XGBRegressor
- RMSE: 5.9375
- R²: 0.9573

Best combined-feature model:

- Model: XGBRegressor
- RMSE: 6.0816
- R²: 0.9552

Conclusion: Physics-informed features improved B5 RMSE relative to raw features.

## Calibration-transfer interpretation

B5 is treated as a held-out target board. Poor B5 performance should not be hidden or optimized away at this stage. If validation on B4 is materially better than testing on B5, this supports the Day 1 visual finding that B5 has a stronger board-specific shift.

Likely causes of B5 transfer difficulty:

1. Board-specific baseline resistance offsets.
2. Sensor-to-sensor response gain differences.
3. Concentration-dependent nonlinear response mismatch.
4. Recovery dynamics that are similar in shape but shifted in scale or slope.

## Saved outputs

- `results/day2/feature_table_methane.csv`
- `results/day2/metrics.csv`
- `results/day2/metrics_regression.csv`
- `results/day2/metrics_classification.csv`
- `results/day2/predictions_b5.csv`
- `results/day2/rf_feature_importance_best_b5.csv`, if RandomForest was trained
- `figures/day2/*.png`

## Implication for Day 3

Day 3 should not introduce deep learning. The next logical step is few-shot target-board adaptation for B5 using a small number of B5 calibration examples, while keeping the same interpretable feature pipeline.
