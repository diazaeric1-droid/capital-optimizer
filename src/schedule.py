"""Quarterly scheduling — lay the selected program into quarters under a per-quarter
rig-day and capex capacity, respecting each project's earliest start. Greedy bin-pack
by capital efficiency (highest-value projects placed first, earliest feasible quarter).
"""
from __future__ import annotations

import pandas as pd

from .projects import Project


def schedule_program(econ: pd.DataFrame, selected_ids: list[str], projects: list[Project],
                     n_quarters: int = 4, rig_per_quarter: float | None = None,
                     capex_per_quarter: float | None = None) -> pd.DataFrame:
    by_id = {p.project_id: p for p in projects}
    sel = (econ[econ["project_id"].isin(selected_ids)]
           .sort_values("capital_efficiency", ascending=False))
    rig_used = {q: 0.0 for q in range(1, n_quarters + 1)}
    capex_used = {q: 0.0 for q in range(1, n_quarters + 1)}
    rows = []
    for _, r in sel.iterrows():
        p = by_id[r["project_id"]]
        placed = None
        for q in range(int(p.earliest_quarter), n_quarters + 1):
            ok_rig = rig_per_quarter is None or rig_used[q] + p.rig_days <= rig_per_quarter
            ok_cap = capex_per_quarter is None or capex_used[q] + p.capex_usd <= capex_per_quarter
            if ok_rig and ok_cap:
                placed = q; break
        if placed is None:                       # capacity full everywhere feasible → last quarter
            placed = n_quarters
        rig_used[placed] += p.rig_days
        capex_used[placed] += p.capex_usd
        rows.append({"project_id": p.project_id, "name": p.name, "category": p.label,
                     "quarter": f"Q{placed}", "capex_usd": p.capex_usd, "rig_days": p.rig_days,
                     "risked_npv_usd": r["risked_npv_usd"]})
    return pd.DataFrame(rows).sort_values(["quarter", "risked_npv_usd"], ascending=[True, False])
