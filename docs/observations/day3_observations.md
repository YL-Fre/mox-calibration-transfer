# Day 3 observations

## Few-shot adaptation substantially improves B5 transfer

Using a very small amount of labeled B5 data significantly reduced transfer error compared with the Day2 zero-shot baseline.

The strongest improvements were obtained using:

* mean/std feature alignment
* simple recalibration
* lightweight RandomForest retraining

This suggests that the dominant domain shift between boards is not complete feature manifold collapse, but rather:

* baseline offset
* gain mismatch
* feature scaling distortion

The PCA results support this interpretation:
B5 initially appears shifted relative to the B1-B3 manifold, but simple statistical alignment substantially restores overlap.

---

## Physics-informed features remain highly beneficial

The physics-informed feature set continued to outperform naive transfer approaches.

In particular:

* normalized response features
* relative resistance metrics
* drift-aware features

appear more stable across boards than raw sensor magnitudes.

This implies that physically meaningful feature engineering reduces the adaptation burden and improves calibration portability.

---

## Mean/std alignment is surprisingly effective

One of the most important findings is that simple statistical feature alignment achieved very large RMSE reductions.

This is scientifically important because it suggests:

* board-to-board variation is structured
* much of the domain shift is low-order statistical deformation
* deep nonlinear adaptation may not always be necessary

This is encouraging for practical deployment because lightweight adaptation is computationally inexpensive and interpretable.

---

## CORAL alignment was unstable

The CORAL-style covariance alignment method did not consistently improve transfer performance.

This likely reflects the fact that MOx sensor drift is not purely Gaussian covariance mismatch.

Possible contributing factors include:

* nonlinear saturation
* heater variation
* hysteresis
* concentration-dependent compression

This negative result is scientifically useful because it suggests that more complex covariance matching alone may not solve MOx calibration transfer.

---

## High-concentration transfer remains difficult

Even after adaptation, high methane concentrations remained the most challenging regime.

Residual errors at high concentration likely arise from:

* sensor saturation
* nonlinear response compression
* reduced sensitivity at high gas loading

This indicates that:

* few-shot adaptation helps substantially
* but saturation-aware modeling is still needed

Future work should investigate:

* nonlinear concentration warping
* concentration-dependent calibration
* adaptive weighting of saturation regions

---

## Important limitation of current dataset

The B5 board contains only a very small number of replicates per concentration.

As a result:

* the current experiments rigorously demonstrate 0-shot vs 1-shot adaptation
* but do not fully validate larger few-shot scaling regimes (2-shot, 5-shot, 10-shot)

Additional replicate measurements would be required for statistically robust scaling analysis.

---

## Practical implication

The overall Day3 results suggest that practical calibration transfer for MOx methane sensing is feasible using only a very small calibration set on the target board.

This is promising for:

* low-cost deployment
* edge sensing systems
* field recalibration
* long-term sensor maintenance

without requiring expensive full-board recalibration procedures.
