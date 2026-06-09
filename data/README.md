# Dataset

This project uses the **Twin Gas Sensor Arrays Dataset** from the UCI Machine Learning Repository.

## Dataset Source

UCI Machine Learning Repository

Dataset:

Twin Gas Sensor Arrays Dataset

https://archive.ics.uci.edu/dataset/361/twin+gas+sensor+arrays

## Why the Data Are Not Included

The original dataset is distributed by UCI and is therefore not stored in this repository.

Users wishing to reproduce the analysis should download the dataset directly from the original source.

## Expected Directory Structure

After downloading, place all raw TXT files into:

```text
data/
├── README.md
└── raw/
    ├── B1_*.txt
    ├── B2_*.txt
    ├── B3_*.txt
    ├── B4_*.txt
    └── B5_*.txt
```

Approximately 640 TXT files are expected.

## Reproducibility

All notebooks and scripts in this repository assume that the raw dataset is located in:

```text
data/raw/
```

No additional preprocessing is required before running the analysis pipeline.
