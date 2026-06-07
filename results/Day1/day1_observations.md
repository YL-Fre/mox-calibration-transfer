
# Day 1 Observations

## Dataset structure

- Total files: 640
- Boards: B1–B5
- Gases:
  - GCO = Carbon Monoxide
  - GEa = Ethanol
  - GEy = Ethylene
  - GMe = Methane

- Each file contains:
  - column 0 = time
  - columns 1–8 = MOx sensor channels

- Sampling interval:
  - 0.01 s

- Approximate experiment duration:
  - 530 s

## Methane response observations

### Board-to-board variation

- Strong baseline offsets exist across boards.
- Response magnitude differs between boards.
- B5 shows noticeably slower recovery behavior.

### Shared temporal structure

Despite amplitude differences, all boards show:

- similar response timing
- similar exposure/recovery phases
- similar dynamic response shape

This suggests that physics-informed temporal features may transfer better than raw signals.

### Transfer learning implications

Raw signal transfer is expected to perform poorly because:

- sensor baselines vary
- absolute amplitudes vary

However, normalized or physically meaningful features may improve transferability.

## Initial conclusion

The dataset appears suitable for:

- calibration transfer
- few-shot adaptation
- physics-informed feature engineering
- sensor-to-sensor generalization studies
