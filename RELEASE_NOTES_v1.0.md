# Release Notes

## Version 1.0 — Initial Public Release

**Release Date:** June 2026

---

## Overview

This is the first public release of the **MOx Calibration Transfer for Combustible Gas Detection** project.

The release contains the complete Day1–Day5 research workflow, including:

* Dataset exploration
* Cross-board calibration transfer benchmarking
* Geometry and manifold analysis
* Few-shot adaptation
* Physics-aware adaptation
* Failure-mode diagnosis

The project investigates how calibration models trained on one set of MOx sensor boards transfer to unseen boards and identifies distinct mechanisms responsible for transfer failure.

---

## Major Findings

### 1. Calibration transfer failures are not uniform

Two qualitatively different failure modes were identified:

* Coverage-limited transfer failure (Board B5)
* Target-intrinsic response compression (Board B1)

These failure modes require different remediation strategies.

### 2. Few-shot adaptation is highly effective for coverage-limited boards

For Board B5, a small number of labeled target samples reduced transfer error by more than 80–90%.

### 3. Physics-informed features improve transfer robustness

Normalized Rs/R₀ and response-magnitude descriptors consistently outperformed raw sensor values across transfer scenarios.

### 4. Complex corrections do not necessarily outperform simple alignment

Global mean/std alignment remained competitive and often superior to more complex physics-aware corrections under realistic calibration budgets.

### 5. Label leakage was discovered and corrected

A feature leakage issue identified during Day4 development was diagnosed and removed. All final conclusions are based on the corrected results.

---

## Repository Contents

This release includes:

* 11 research notebooks
* Reproducible Python analysis scripts
* Observation documents
* Publication-quality figures
* Failure-mode analysis framework

---

## Known Limitations

* Analysis focuses primarily on methane transfer performance.
* Evaluation is based on a single public benchmark dataset.
* Deep learning approaches were intentionally excluded.

---

## Future Directions

Potential future work includes:

* Physics-informed domain adaptation
* Sensor drift compensation
* Transfer learning across sensor generations
* Topological Data Analysis (TDA)
* Active calibration strategies

---

## Release Status

**Version:** 1.0

**Status:** Stable

**Experimental Phase:** Complete

**Public Repository Release:** Yes
