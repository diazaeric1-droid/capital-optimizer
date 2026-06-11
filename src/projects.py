"""Project inventory — the capital backlog the optimizer chooses from.

Each project is a discrete capital opportunity (a new drill, a DUC completion, a
recompletion, a workover, or an artificial-lift conversion) with a type curve, a
capex, a chance of success, and the rig-days it consumes. Production deployments
would load this from the planning system (ARIES/PHDWin scenarios, a DUC tracker,
the workover backlog); the CSV contract here mirrors that shape.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union

import pandas as pd

CATEGORIES = ("new_drill", "duc_completion", "recompletion", "workover", "alift_conversion")
CATEGORY_LABEL = {
    "new_drill": "New drill", "duc_completion": "DUC completion",
    "recompletion": "Recompletion", "workover": "Workover",
    "alift_conversion": "Artificial-lift conversion",
}

# Ordered list of required CSV column names (matches Project dataclass fields).
REQUIRED_CSV_COLUMNS: list[str] = [
    "project_id", "name", "category", "area",
    "capex_usd", "qi_bopd", "di_annual", "b",
    "opex_per_bbl", "nri", "pc", "rig_days", "earliest_quarter",
]


@dataclass
class Project:
    project_id: str
    name: str
    category: str
    area: str
    capex_usd: float
    qi_bopd: float          # initial incremental oil rate of the add
    di_annual: float        # Arps nominal annual decline
    b: float                # Arps hyperbolic exponent (0 = exponential)
    opex_per_bbl: float
    nri: float              # net revenue interest (operator's share of revenue)
    pc: float               # chance of (technical/commercial) success, 0-1
    rig_days: float         # rig/crew days the project consumes
    earliest_quarter: int   # earliest quarter it can start (1-4)

    @property
    def label(self) -> str:
        return CATEGORY_LABEL.get(self.category, self.category)


def load_projects(path: str | Path) -> list[Project]:
    df = pd.read_csv(path)
    keys = {f.name for f in fields(Project)}
    out = []
    for _, row in df.iterrows():
        out.append(Project(**{k: row[k] for k in keys}))
    return out


def projects_from_csv(path: Union[str, Path]) -> list[Project]:
    """Load a list of Project objects from a user-supplied CSV file.

    Validates that all required columns are present and that the DataFrame
    is non-empty. Raises ``ValueError`` with a descriptive message on any
    validation failure so callers (e.g. the Streamlit app) can surface a clean
    ``st.error`` rather than an unhandled exception.

    Column contract (all required):
        project_id, name, category, area, capex_usd, qi_bopd, di_annual, b,
        opex_per_bbl, nri, pc, rig_days, earliest_quarter

    Type coercion mirrors ``load_projects``:
        - Numeric columns are coerced via pandas (non-numeric → NaN → error).
        - ``earliest_quarter`` is cast to int.
        - ``project_id``, ``name``, ``category``, ``area`` are kept as str.
    """
    df = pd.read_csv(path)

    # --- column presence check -----------------------------------------------
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Required: {REQUIRED_CSV_COLUMNS}"
        )

    if df.empty:
        raise ValueError("CSV contains no data rows.")

    # --- numeric coercion + NaN check ----------------------------------------
    numeric_cols = [
        "capex_usd", "qi_bopd", "di_annual", "b",
        "opex_per_bbl", "nri", "pc", "rig_days", "earliest_quarter",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_rows = df[numeric_cols].isna().any(axis=1)
    if bad_rows.any():
        raise ValueError(
            f"Non-numeric or missing values in numeric columns at row(s): "
            f"{list(df.index[bad_rows] + 1)}."
        )

    # Cast earliest_quarter to int (it arrives as float after to_numeric).
    df["earliest_quarter"] = df["earliest_quarter"].astype(int)

    keys = {f.name for f in fields(Project)}
    return [Project(**{k: row[k] for k in keys}) for _, row in df.iterrows()]


def projects_to_frame(projects: list[Project]) -> pd.DataFrame:
    return pd.DataFrame([p.__dict__ for p in projects])
