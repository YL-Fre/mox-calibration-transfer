# Day 3 observations

## Objective

Day3 investigated whether a small number of labeled target-board samples can correct board-to-board transfer error.

This experiment focused on **B5**, which Day2+ identified as a representative example of a **coverage-limited transfer failure**.

The central question was:

> Can a small amount of target-board calibration data rescue transfer performance without requiring complete retraining?

---

# Key Finding

Few-shot adaptation dramatically improves transfer performance.

Across multiple adaptation strategies, only a small number of labeled B5 samples were required to substantially reduce transfer error.

The strongest improvements were obtained using:

* mean/std feature alignment
* simple linear recalibration
* lightweight RandomForest retraining

The best methods reduced transfer RMSE from approximately:

* ~6 ppm (zero-shot)
* to ~1 ppm (few-shot adaptation)

This demonstrates that calibration transfer can be recovered with very limited target-board supervision.

---

# Scientific Interpretation

## Day2+ Context

Day2+ revealed two distinct transfer failure modes:

### Failure Mode A — Coverage-Limited

Target:

* B5

Characteristics:

* transfer error strongly correlated with source manifold coverage
* transfer error decreases as source diversity increases
* geometric coverage metrics predict transferability

Interpretation:

The source domain does not sufficiently represent the target board.

---

### Failure Mode B — Target-Intrinsic

Target:

* B1

Characteristics:

* severe high-concentration compression
* systematic underprediction at 80–100 ppm
* weak relationship between transfer error and source coverage

Interpretation:

The transfer difficulty appears to originate from the target response characteristics themselves.

---

## What Day3 Demonstrates

Day3 directly tested whether a coverage-limited failure can be corrected through target-side adaptation.

The answer is clearly:

> Yes.

Simple adaptation methods almost completely eliminate the B5 transfer penalty.

This result suggests that:

* the source manifold already contains most of the relevant information
* the remaining domain shift is relatively low-order
* adaptation mainly corrects statistical mismatch rather than discovering new structure

In practical terms:

B5 is difficult because it is insufficiently represented, not because it is fundamentally incompatible with the source boards.

---

# Mean/Std Alignment Is Surprisingly Effective

One of the most important findings of Day3 is the effectiveness of simple statistical alignment.

Mean/std alignment consistently achieved some of the best results despite its simplicity.

This suggests that a substantial portion of board-to-board variation can be explained by:

* baseline offset
* gain mismatch
* feature scaling distortion

rather than complex nonlinear domain shifts.

The PCA visualizations support this interpretation.

Before adaptation:

* B5 occupies a shifted region of feature space.

After adaptation:

* B5 overlaps substantially more with the source-board manifold.

This indicates that lightweight alignment may be sufficient for many practical deployment scenarios.

---

# Physics-Informed Features Remain Valuable

The physics-informed feature set continued to demonstrate strong transferability.

In particular:

* normalized response features
* relative resistance metrics
* drift-aware features

appear more stable across boards than raw signal magnitudes.

This reinforces a central theme of the project:

> Physically meaningful features reduce the adaptation burden.

The better the feature representation captures sensor physics, the less adaptation is required downstream.

---

# CORAL Alignment Was Unstable

CORAL-style covariance alignment did not consistently improve transfer performance.

In some cases it produced unstable or degraded results.

This suggests that MOx sensor domain shift cannot be fully described as a covariance-matching problem.

Potential causes include:

* nonlinear saturation
* heater variability
* adsorption/desorption hysteresis
* concentration-dependent response compression

This negative result is scientifically useful because it narrows the class of adaptation mechanisms likely to succeed.

---

# High-Concentration Transfer Remains the Most Challenging Regime

Although overall transfer performance improved substantially, high methane concentrations remained the most difficult operating region.

Residual errors at high concentration are consistent with:

* sensor saturation
* nonlinear response compression
* reduced sensitivity at high gas loading

Few-shot adaptation reduces these errors, but does not fully eliminate the underlying physical nonlinearity.

Future work should investigate:

* nonlinear concentration warping
* concentration-aware calibration
* saturation-aware loss functions
* concentration-dependent adaptation

---

# Dataset Limitation

The B5 dataset contains only a small number of replicate measurements per concentration.

Therefore:

* the distinction between 0-shot and few-shot adaptation is clearly demonstrated
* but scaling behaviour for larger shot counts cannot be rigorously characterized

Additional replicates would be required for statistically robust adaptation-scaling analysis.

---

# Practical Implications

The Day3 results suggest that practical calibration transfer for MOx methane sensing is achievable with only a very small calibration set collected on the target board.

This is promising for:

* low-cost deployment
* edge sensing systems
* field recalibration
* long-term sensor maintenance

without requiring complete board-specific retraining.

---

# Most Important Conclusion

Day3 establishes that:

> Coverage-limited transfer failures can be effectively rescued through few-shot adaptation.

However, Day3 does **not** answer the most important question raised by Day2+:

> Can few-shot adaptation rescue B1?

B1 exhibited a fundamentally different failure mode characterized by high-concentration response compression and systematic underprediction.

Whether this target-intrinsic failure can be corrected through adaptation remains unknown.

This directly motivates the next stage of the project:

# Day3.5 — Can Few-Shot Adaptation Rescue B1?
