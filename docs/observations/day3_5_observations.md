\# Day3.5 Observations



\## Can Few-Shot Adaptation Rescue B1?



\---



\## Objective



Day2+ identified two qualitatively different transfer failure modes:



\### B5



A coverage-limited transfer failure characterized by poor source-target manifold overlap.



\### B1



A target-intrinsic failure characterized by systematic response compression at high methane concentrations.



Day3 demonstrated that few-shot adaptation can substantially improve transfer performance on B5.



The goal of Day3.5 was therefore to determine whether the same adaptation strategies can also correct the B1 anomaly.



\---



\# Experimental Setup



Source boards:



```text

B2

B3

B4

B5

```



Target board:



```text

B1

```



Adaptation methods:



```text

mean\_std\_alignment

linear\_recalibration

residual\_ridge

rf\_retraining

rf\_weighted\_retraining

```



Few-shot levels:



```text

0-shot

1-shot

2-shot

5-shot

10-shot

```



Evaluation focused on:



```text

Overall RMSE

High-concentration RMSE (80–100 ppm)

Signed prediction error

Prediction-vs-true concentration behavior

```



\---



\# Finding 1



\## Few-shot adaptation improves B1 transfer performance



Without adaptation:



```text

Overall RMSE ≈ 15.4 ppm

```



Best adaptation result:



```text

rf\_weighted\_retraining

10-shot

RMSE ≈ 11.3 ppm

```



Relative improvement:



```text

≈ 27%

```



This demonstrates that target-side labels contain useful information and that B1 is not completely irrecoverable.



However, the magnitude of improvement is substantially smaller than what was observed for B5.



\---



\# Finding 2



\## High-concentration performance improves but remains difficult



The largest errors occur at:



```text

80 ppm

90 ppm

100 ppm

```



Without adaptation:



```text

High-concentration RMSE ≈ 21 ppm

```



Best adaptation result:



```text

≈ 12 ppm

```



Improvement:



```text

≈ 40–45%

```



This confirms that few-shot adaptation partially corrects the B1 anomaly.



However, substantial error remains even after adaptation.



Unlike B5, the transfer problem is not completely resolved by adding a small number of labeled target samples.



\---



\# Finding 3



\## High-concentration response compression persists



Day2+ identified a characteristic pattern:



```text

Low concentrations:

overestimation



High concentrations:

underestimation

```



Most notably:



```text

100 ppm



predicted ≈ 75–85 ppm

```



in the zero-shot setting.



After adaptation:



```text

high-concentration predictions move closer

to the ideal calibration line

```



but remain systematically below ground truth.



Signed-error analysis shows:



```text

100 ppm



before adaptation:

≈ -25 ppm



after adaptation:

≈ -17 ppm

```



Thus:



```text

compression is reduced

but not eliminated

```



\---



\# Finding 4



\## B1 behaves fundamentally differently from B5



Day3 showed:



```text

B5 transfer error

↓ dramatically

with only a few target labels

```



suggesting that B5 was primarily a coverage problem.



In contrast:



```text

B1 transfer error

↓ only partially

```



even after adaptation.



This suggests:



```text

B5:

coverage-limited



B1:

target-intrinsic

```



represent distinct failure mechanisms.



\---



\# Scientific Interpretation



The evidence collected across Day2+, Day3, and Day3.5 supports the following picture:



\### B5



```text

Source manifold coverage is insufficient.



Few-shot adaptation successfully fills

the missing regions of feature space.



Large nonlinear correction is unnecessary.

```



\### B1



```text

Target response behavior differs

from the source boards.



The anomaly cannot be explained solely

by source-target coverage mismatch.



Target-specific response compression

persists even after adaptation.

```



Therefore:



```text

B1 remains a target-intrinsic failure mode.

```



\---



\# Failure Mode Validation



| Target | Failure Mode     | 0-shot RMSE (ppm) | Best RMSE (ppm) | Relative Improvement |

| ------ | ---------------- | ----------------: | --------------: | -------------------: |

| B1     | Target-intrinsic |             \~15.4 |           \~11.3 |                 \~27% |

| B5     | Coverage-limited |             \~24.4 |            \~1–2 |                 >90% |



The contrast is substantial.



B5 becomes nearly solved after adaptation.



B1 remains challenging.



\---



\# Answers to the Day3.5 Questions



\## 1. Can few-shot adaptation rescue B1?



Partially.



Adaptation significantly improves performance but does not fully remove the B1 anomaly.



\---



\## 2. Is B1 fundamentally different from B5?



Yes.



B5 behaves like a coverage-limited transfer failure.



B1 behaves like a target-intrinsic transfer failure.



\---



\## 3. Does B1 remain target-intrinsic after adaptation?



Yes.



The persistence of high-concentration compression after adaptation indicates that the underlying response behavior of B1 differs from the source boards.



\---



\## 4. What is the next scientific question?



The next question is:



```text

Can physics-aware nonlinear calibration

correct the residual high-concentration

response compression observed on B1?

```



Potential directions include:



```text

Piecewise calibration



Monotonic regression



Weighted high-concentration loss



Sensor saturation modeling



Board-specific nonlinear correction

```



\---



\# Key Takeaway



Day3.5 demonstrates that few-shot adaptation partially rescues B1 but does not fully eliminate its characteristic high-concentration compression.



Combined with Day2+ and Day3, the evidence suggests that MOx calibration transfer contains at least two distinct transfer failure modes:



1\. Coverage-limited failures that are highly adaptation-responsive (B5)



2\. Target-intrinsic failures that remain difficult even after adaptation (B1)



Understanding and correcting the second category is the next major challenge of the project.



