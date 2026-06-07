# Day5 Observations — B1 vs B5 Failure-Mode Analysis

## Purpose

Day5 is a mechanism-analysis step, not a leaderboard-optimization step. The goal is to compare B1 and B5 as potentially different transfer-failure modes using lightweight, leakage-safe analysis.

## Leakage and memory-safety notes

- Forbidden label/metadata columns were excluded from model input features.
- The final leakage check is saved to `results/day5/day5_debug_feature_leakage_check.csv`.
- The analysis intentionally avoids deep learning, UMAP/t-SNE, nested cross-validation, large grid search, and repeated permutation importance.
- PCA uses aggregated board/concentration medians for compact trajectory analysis.
- All figures are saved and closed after creation.

## Representative features used for response-curve and gain/curvature analysis

- `s2_phys_minimum_response`
- `s2_phys_rs_r0_min`
- `s2_phys_normalized_response`
- `s2_phys_response_magnitude`
- `s1_phys_rs_r0_min`
- `s1_phys_normalized_response`

## 1. How does B1 differ from B5?

B1 and B5 should not be treated as the same transfer problem. The diagnostic table below summarizes their relative behavior:

| board   |   coverage_limited_score |   gain_mismatch_score |   curvature_mismatch_score |   intrinsic_deformation_score |   high_concentration_compression_score | adaptation_responsiveness                                     | likely_failure_mode                                  |
|:--------|-------------------------:|----------------------:|---------------------------:|------------------------------:|---------------------------------------:|:--------------------------------------------------------------|:-----------------------------------------------------|
| B1      |                0.0409976 |             0.0102887 |                  0.0648018 |                     0.0533556 |                              0.0210098 | partial in prior Day3.5 evidence; persistent residuals likely | target-intrinsic deformation / compression candidate |
| B5      |                0.137007  |             0.0415025 |                  0.0545716 |                     0.0914277 |                              0.0651445 | high in prior Day3 evidence; not re-proven by Day5            | mixed coverage and deformation evidence              |

Current evidence supports the working interpretation that B5 is more consistent with a coverage-limited or shifted-domain problem, whereas B1 shows stronger evidence of intrinsic response deformation or high-concentration compression. This remains a cautious interpretation, not proof of an exact physical cause.

## 2. Is B1 mainly gain mismatch, curvature mismatch, or compression?

Day5 decomposes feature response into linear gain, quadratic curvature, and high-concentration residual terms. If B1 has a large curvature/high-concentration score relative to its global gain score, it should be interpreted as deformation/compression rather than a simple gain offset. The physical mechanism remains unresolved.

## 3. Is B5 primarily coverage-limited?

The Day5 evidence should be read together with prior Day3 evidence: B5 was adaptation-responsive under mean/std alignment, suggesting much of its error can be explained by statistical mismatch or limited target coverage. Day5 does not re-prove adaptation responsiveness; it checks whether B5 lies closer to source-like trajectories and feature relationships than B1.

## 4. Which features appear board-invariant?

Top board-invariant candidate features from the transferability score:

- `s2_phys_rs_r0_min`
- `s2_phys_normalized_response`
- `s1_phys_normalized_response`
- `s1_phys_rs_r0_min`
- `s2_phys_minimum_response`
- `s1_phys_minimum_response`
- `s1_raw_min`
- `s1_raw_q10`

These are features with relatively low source-board coefficient of variation and relatively monotonic concentration response.

## 5. Which features fail specifically on B1?

Features with the largest B1-specific deviation relative to B5:

- `s1_phys_response_magnitude`
- `s1_phys_rs_r0_min`
- `s1_phys_normalized_response`
- `duration_s`
- `s1_phys_rs_r0_final`
- `s2_phys_rs_r0_final`
- `s1_raw_std`
- `s2_phys_rs_r0_min`

These features should be inspected as candidates for B1-specific high-concentration instability or compression.

## 6. Does PCA support two different failure modes?

The PCA trajectory analysis uses leakage-free features only. Explained variance: PC1=34.2%, PC2=26.0%. If B5 lies outside the source-board coverage but follows a similar concentration direction, that supports a coverage-limited interpretation. If B1 bends, compresses, or deviates at high concentration in PCA space, that supports an intrinsic deformation interpretation.

## 7. What should Day6 investigate?

Day6 should investigate whether B1's apparent response deformation can be corrected by feature-level transformations that explicitly model saturation/compression without label leakage. Recommended next steps:

1. Compare sensor-wise saturation indicators and dynamic response features for B1 versus source boards.
2. Test monotonic calibration mappings fitted only on small target calibration subsets.
3. Separate baseline drift, gain shift, and high-concentration saturation using sensor-level physics features.
4. Validate whether B1-like compression appears in any non-methane gases or is methane-specific.
5. Keep Day6 memory-safe and mechanism-focused rather than running large model searches.

## Scientific caution

Day5 supports the hypothesis that B1 and B5 represent two different transfer failure modes. It does not prove the microscopic physical cause of B1 behavior. The current evidence supports response deformation/compression, but additional controlled analysis is required.
