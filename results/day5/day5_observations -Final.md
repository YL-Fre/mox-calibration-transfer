# Day5 Observations

## Understanding Transfer Failure Modes in MOx Methane Calibration Transfer

### Objective

The goal of Day5 was not to develop a better adaptation algorithm, but rather to understand the mechanisms underlying board-to-board transfer failures.

Previous experiments suggested that:

* B5 transfer errors could be substantially reduced using simple few-shot adaptation.
* B1 transfer errors remained more difficult to correct even after adaptation.
* Physics-aware correction methods did not clearly outperform simple statistical alignment.

These observations motivated the central Day5 question:

> Why do different boards exhibit different transfer behaviors?

---

# 1. Board-Level Response Curves

Median response curves were compared across all methane concentrations using leakage-free physics-informed features.

Several observations emerged:

* Most response features remained strongly monotonic with methane concentration.
* B1 generally followed the same concentration-response trend as source boards.
* B5 often exhibited larger offsets and gain differences relative to source boards.
* No universally obvious saturation pattern unique to B1 was observed across all selected features.

For features such as:

* s2_phys_rs_r0_min
* s1_phys_rs_r0_min
* normalized response metrics

B1 remained visually similar to the source-board manifold.

In contrast, B5 frequently showed larger displacement from source-board trajectories.

### Interpretation

The response-curve analysis suggests that B5 exhibits larger global response mismatch than B1.

This result is somewhat counterintuitive given the adaptation results from Day3 and Day3.5.

---

# 2. Gain–Curvature Decomposition

Linear and quadratic concentration-response models were fitted for selected stable features.

Metrics extracted:

* Linear gain coefficient
* Quadratic curvature coefficient
* High-concentration residual behavior

Results showed:

* Gain differences between boards were relatively modest.
* Curvature differences existed but were generally small.
* The largest high-concentration residual signatures were often associated with B5 rather than B1.

### Interpretation

Day5 did not find strong evidence that B1 can be explained purely by gain mismatch.

Likewise, no single curvature metric fully explained the persistent adaptation difficulty observed for B1.

Instead, gain and curvature alone appear insufficient to characterize transfer difficulty.

---

# 3. PCA Feature-Space Trajectories

PCA was performed using leakage-free physics-informed features.

The first two principal components explained approximately:

* PC1: 34%
* PC2: 26%

of total variance.

Board trajectories were traced across methane concentration levels.

### Observations

B1:

* Followed a smooth trajectory.
* Maintained monotonic progression with concentration.
* Remained close to the overall source-board manifold.

B3 and B4:

* Showed similar geometric behavior.

B5:

* Displayed the largest trajectory displacement.
* Exhibited stronger lateral movement in PCA space.
* Showed trajectory bending and local geometric distortion.

### Interpretation

The PCA results indicate that B5 is geometrically farther from the source-board manifold than B1.

This observation is important because it demonstrates:

> Geometric distance alone does not determine transfer difficulty.

Despite being farther from the source manifold, B5 was previously shown to be substantially more adaptation-responsive.

---

# 4. Feature Transferability Analysis

Transferability scores were computed using:

* Between-board stability
* Monotonicity
* Deviation from source-board median
* High-concentration deviation

Several features exhibited strong board invariance:

### Most stable features

* s2_phys_rs_r0_min
* s2_phys_normalized_response
* s1_phys_normalized_response
* s1_phys_rs_r0_min

These features remained:

* monotonic,
* physically interpretable,
* relatively robust across boards.

### Less transferable features

Features based on:

* response magnitude,
* minimum response,
* raw sensor statistics

showed larger board-specific deviations.

### Interpretation

Physics-informed normalized features remain the strongest candidates for robust calibration-transfer pipelines.

---

# 5. B1 versus B5 Failure Modes

A diagnostic comparison was performed using:

* coverage-limited score
* intrinsic deformation score
* high-concentration compression score

Unexpectedly:

* B5 scored higher than B1 on all three diagnostic measures.
* B5 exhibited larger manifold displacement.
* B5 exhibited larger residual structure.
* B5 exhibited stronger high-concentration instability.

However, previous adaptation experiments demonstrated:

* B5 remains substantially correctable.
* B1 remains comparatively resistant to correction.

### Key Insight

Day5 suggests that:

> Transfer difficulty is not determined solely by the magnitude of distribution shift.

Instead:

B5 appears to represent:

* a larger but largely correctable distribution shift.

B1 appears to represent:

* a smaller geometric deviation,
* but one that is less responsive to adaptation.

Thus:

> Distance from the source manifold and adaptation responsiveness are not equivalent concepts.

---

# 6. Scientific Interpretation

Across Days 2–5, the evidence supports the following view:

### B5

* Larger global deviation from source boards.
* Stronger geometric displacement in feature space.
* Larger residual signatures.
* Nevertheless highly adaptation-responsive.

### B1

* Smaller overall deviation.
* Relatively smooth PCA trajectory.
* Limited evidence for extreme saturation.
* Remains difficult to fully rescue using simple adaptation methods.

Therefore:

> B1 and B5 should not be treated as the same transfer problem.

Different transfer failures may arise from fundamentally different response mechanisms.

The precise physical mechanism responsible for B1 remains unresolved.

Current evidence supports the existence of board-specific response deformation, but does not prove a particular physical cause.

---

# Final Conclusion

Day5 successfully shifted the project from performance optimization toward mechanism understanding.

The original hypothesis that:

* B5 is purely coverage-limited,
* B1 is purely intrinsic deformation,

was not fully supported by the data.

Instead, the results reveal a more nuanced picture:

* B5 is farther from the source manifold but easier to adapt.
* B1 is closer to the source manifold but harder to adapt.

This suggests that successful calibration transfer depends not only on the magnitude of domain shift, but also on the structure of that shift.

Understanding transferability therefore requires analyzing both:

* geometric distance,
* and adaptation responsiveness.

This represents the primary scientific outcome of the project.

---

## Project Outcome (Days 1–5)

The five-day study established:

1. Board-to-board transfer is a significant challenge for MOx methane sensing.
2. Simple few-shot adaptation can substantially improve transfer performance.
3. Physics-aware correction alone does not guarantee better transfer.
4. Different boards exhibit qualitatively different transfer behaviors.
5. Transferability cannot be explained solely by manifold distance.

These findings provide a foundation for future work on robust calibration transfer for MOx gas sensors.
