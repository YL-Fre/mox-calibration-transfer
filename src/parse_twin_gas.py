"""Utilities for inspecting and parsing UCI #361 Twin Gas Sensor Arrays files.

This module is intentionally conservative because the archive filename layout
should be verified during Day 1 before hard-coding assumptions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


GAS_ALIASES = {
    "methane": "methane",
    "ch4": "methane",
    "ethanol": "ethanol",
    "ethylene": "ethylene",
    "co": "carbon_monoxide",
    "carbonmonoxide": "carbon_monoxide",
    "carbon_monoxide": "carbon_monoxide",
}


@dataclass(frozen=True)
class TrialMeta:
    path: Path
    board: str | None
    gas: str | None
    concentration_ppm: float | None
    day: int | None


def list_data_files(raw_dir: Path, suffixes: tuple[str, ...] = (".txt", ".csv", ".dat")) -> list[Path]:
    return sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def infer_metadata_from_name(path: Path) -> TrialMeta:
    """Best-effort metadata extraction from filename.

    Day 1 task: compare this output against the official archive docs and adjust
    patterns if needed. We avoid assuming one exact naming convention.
    """
    name = path.stem.lower()

    board = None
    m = re.search(r"(?:board|unit|b)[_\- ]?([1-5])", name)
    if m:
        board = f"B{m.group(1)}"

    gas = None
    compact = name.replace("-", "_").replace(" ", "_")
    for token, canonical in GAS_ALIASES.items():
        if token in compact:
            gas = canonical
            break

    concentration_ppm = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:ppm|ppb)?", name)
    if m:
        concentration_ppm = float(m.group(1))

    day = None
    m = re.search(r"(?:day|d)[_\- ]?(\d+)", name)
    if m:
        day = int(m.group(1))

    return TrialMeta(path=path, board=board, gas=gas, concentration_ppm=concentration_ppm, day=day)


def read_trial(path: Path) -> pd.DataFrame:
    """Read one trial file with flexible delimiter handling.

    Expected raw signals are time-series sensor conductance/resistance values.
    The exact column names are assigned after inspecting the archive.
    """
    df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
    df = df.dropna(axis=1, how="all")
    return df


def summarize_files(files: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for path in files:
        meta = infer_metadata_from_name(path)
        rows.append({
            "path": str(path),
            "filename": path.name,
            "board": meta.board,
            "gas": meta.gas,
            "concentration_ppm": meta.concentration_ppm,
            "day": meta.day,
            "size_mb": path.stat().st_size / 1e6,
        })
    return pd.DataFrame(rows)


def add_time_column(df: pd.DataFrame, sampling_hz: int = 100) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "time_s", np.arange(len(out)) / sampling_hz)
    return out
