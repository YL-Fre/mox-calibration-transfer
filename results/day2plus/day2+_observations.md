Day2+ Observations
Observation 1 — Calibration transfer performance varies strongly across target boards

Transfer RMSE differs substantially depending on the target board.

Among all targets, B1 consistently exhibits the largest transfer error, while B5 shows intermediate difficulty and B2/B3 remain comparatively easier.

This indicates that calibration transfer is not determined solely by source-domain quality; target-specific characteristics also play an important role.

Observation 2 — Source-domain diversity improves robustness but does not universally solve transfer failure

Increasing the number of source boards generally reduces average RMSE and improves robustness across targets.

However, the improvement is target-dependent.

For some targets (especially B5), additional source diversity substantially improves performance, whereas for B1 the improvement is limited.

Therefore, source diversity is beneficial but not sufficient to guarantee successful transfer.

Observation 3 — Geometry similarity is an inconsistent predictor of transferability

PCA-based centroid distance and covariance mismatch show weak or inconsistent relationships with transfer RMSE when evaluated across all transfer scenarios.

This suggests that simple geometric similarity between source and target manifolds is insufficient to explain calibration-transfer success.

Observation 4 — Coverage metrics predict transferability for B5

For target B5, source-domain coverage metrics show strong negative correlations with transfer RMSE.

Examples include:

source volume (95% variance)
effective dimensionality
high-concentration coverage

Larger source-domain coverage is associated with lower transfer error.

This supports a coverage-limited transfer mechanism for B5.

Observation 5 — B1 exhibits concentration-dependent response compression

B1 shows a distinct failure pattern.

Prediction-versus-truth analysis reveals systematic underestimation at high methane concentrations.

The signed error becomes increasingly negative with concentration and reaches approximately −15 ppm near 100 ppm methane.

This behavior is observed regardless of source-board selection.

Observation 6 — B1 transfer failure is primarily target-intrinsic

Unlike B5, neither geometry metrics nor coverage metrics strongly predict B1 transfer performance.

The dominant error source is a target-specific response compression at high concentration rather than insufficient source-domain coverage.

This indicates a target-intrinsic transfer limitation.

Observation 7 — Two distinct transfer-failure modes emerge

The results support the existence of two different transfer mechanisms:

Coverage-limited transfer

Representative target:

B5

Characteristics:

sensitive to source diversity
predictable using coverage metrics

Engineering implication:

increase source-board diversity
Target-intrinsic transfer failure

Representative target:

B1

Characteristics:

strong high-concentration bias
weak dependence on source selection

Engineering implication:

apply target-side adaptation
(few-shot calibration or domain adaptation)
Day2+ Conclusion

Calibration transfer difficulty cannot be explained by geometric similarity alone.

Some targets (e.g., B5) are primarily limited by source-domain coverage, while others (e.g., B1) exhibit target-specific nonlinear response behavior that cannot be corrected through source selection alone.

These findings motivate the transition to Day3, where target-side few-shot adaptation will be investigated as a mechanism for correcting target-intrinsic transfer failure.