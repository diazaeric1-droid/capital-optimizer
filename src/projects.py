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

import pandas as pd

CATEGORIES = ("new_drill", "duc_completion", "recompletion", "workover", "alift_conversion")
CATEGORY_LABEL = {
    "new_drill": "New drill", "duc_completion": "DUC completion",
    "recompletion": "Recompletion", "workover": "Workover",
    "alift_conversion": "Artificial-lift conversion",
}


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


def projects_to_frame(projects: list[Project]) -> pd.DataFrame:
    return pd.DataFrame([p.__dict__ for p in projects])
