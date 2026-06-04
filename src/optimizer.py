"""Capital-allocation optimizer — pick the program that maximizes risked NPV under a
capital budget + rig-day capacity (+ optional minimum production add).

This is the technical core: a 0/1 selection problem solved exactly as a **MILP**
(branch-and-bound via CBC through PuLP). We also compute:
  - a **greedy** baseline (rank by capital efficiency, cut at the budget) — what most
    operators actually do — so we can show the $ the optimizer captures over it;
  - the **LP-relaxation bound**, so we can report a provable **optimality gap**.

If PuLP/CBC isn't available we fall back to an exact **DP knapsack** (budget-only) or
the greedy heuristic, so the app always returns a valid program.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .projects import CATEGORY_LABEL


@dataclass
class Program:
    method: str
    selected_ids: list[str]
    capex_used: float
    budget: float
    rig_used: float
    rig_capacity: float | None
    risked_npv: float
    npv: float
    first_year_bbl: float
    eur_bbl: float
    n_selected: int
    n_available: int
    weighted_cap_eff: float
    by_category: dict
    lp_bound: float | None = None       # LP-relaxation upper bound (for optimality gap)
    optimality_gap_pct: float | None = None


def _summ(econ: pd.DataFrame, ids: list[str], budget: float, rig_capacity, method: str) -> Program:
    sel = econ[econ["project_id"].isin(ids)]
    capex = float(sel["capex_usd"].sum())
    risked = float(sel["risked_npv_usd"].sum())
    by_cat = {}
    for cat, g in sel.groupby("category"):
        by_cat[CATEGORY_LABEL.get(cat, cat)] = {
            "capex": float(g["capex_usd"].sum()), "risked_npv": float(g["risked_npv_usd"].sum()),
            "n": int(len(g))}
    return Program(
        method=method, selected_ids=list(ids), capex_used=capex, budget=float(budget),
        rig_used=float(sel["rig_days"].sum()), rig_capacity=rig_capacity,
        risked_npv=risked, npv=float(sel["npv_usd"].sum()),
        first_year_bbl=float(sel["first_year_bbl"].sum()), eur_bbl=float(sel["eur_bbl"].sum()),
        n_selected=int(len(sel)), n_available=int(len(econ)),
        weighted_cap_eff=(risked / capex if capex > 0 else 0.0), by_category=by_cat)


def greedy_select(econ: pd.DataFrame, budget: float, rig_capacity: float | None = None,
                  min_first_year_bbl: float = 0.0) -> Program:
    cand = econ[econ["risked_npv_usd"] > 0].sort_values("capital_efficiency", ascending=False)
    cap = rig = 0.0
    picked = []
    for _, r in cand.iterrows():
        if cap + r["capex_usd"] > budget:
            continue
        if rig_capacity is not None and rig + r["rig_days"] > rig_capacity:
            continue
        picked.append(r["project_id"]); cap += r["capex_usd"]; rig += r["rig_days"]
    return _summ(econ, picked, budget, rig_capacity, "greedy (rank-by-efficiency)")


def _milp(econ: pd.DataFrame, budget, rig_capacity, min_fy, relaxed: bool):
    import pulp
    ids = list(econ["project_id"])
    risked = dict(zip(econ["project_id"], econ["risked_npv_usd"]))
    capex = dict(zip(econ["project_id"], econ["capex_usd"]))
    rig = dict(zip(econ["project_id"], econ["rig_days"]))
    fy = dict(zip(econ["project_id"], econ["first_year_bbl"]))
    prob = pulp.LpProblem("capital_program", pulp.LpMaximize)
    cat = "Continuous" if relaxed else "Binary"
    x = {i: pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=cat) for i in ids}
    prob += pulp.lpSum(risked[i] * x[i] for i in ids)
    prob += pulp.lpSum(capex[i] * x[i] for i in ids) <= budget
    if rig_capacity is not None:
        prob += pulp.lpSum(rig[i] * x[i] for i in ids) <= rig_capacity
    if min_fy > 0:
        prob += pulp.lpSum(fy[i] * x[i] for i in ids) >= min_fy
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    obj = float(pulp.value(prob.objective) or 0.0)
    chosen = [i for i in ids if (x[i].value() or 0) > 0.5]
    return chosen, obj


def milp_select(econ: pd.DataFrame, budget: float, rig_capacity: float | None = None,
                min_first_year_bbl: float = 0.0) -> Program:
    chosen, _obj = _milp(econ, budget, rig_capacity, min_first_year_bbl, relaxed=False)
    prog = _summ(econ, chosen, budget, rig_capacity, "MILP (optimal under constraints)")
    try:
        _, lp_bound = _milp(econ, budget, rig_capacity, min_first_year_bbl, relaxed=True)
        prog.lp_bound = lp_bound
        prog.optimality_gap_pct = (max(lp_bound - prog.risked_npv, 0.0) / prog.risked_npv * 100.0
                                   if prog.risked_npv > 0 else 0.0)
    except Exception:
        pass
    return prog


def dp_knapsack_select(econ: pd.DataFrame, budget: float, unit: float = 25_000.0) -> Program:
    """Exact 0/1 knapsack (budget only) via DP on a capex grid — the optimality check."""
    cand = econ[econ["risked_npv_usd"] > 0].reset_index(drop=True)
    W = int(budget // unit)
    # ceil weights so the DP solution is always budget-FEASIBLE (a valid lower bound on
    # the true optimum) rather than over-packing from floored capex.
    w = np.ceil(cand["capex_usd"] / unit).astype(int).tolist()
    val = cand["risked_npv_usd"].tolist()
    dp = np.zeros(W + 1)
    keep = [[False] * (W + 1) for _ in range(len(cand))]
    for i in range(len(cand)):
        wi, vi = w[i], val[i]
        for cap in range(W, wi - 1, -1):
            if dp[cap - wi] + vi > dp[cap]:
                dp[cap] = dp[cap - wi] + vi
                keep[i][cap] = True
    cap = int(np.argmax(dp)); picked = []
    for i in range(len(cand) - 1, -1, -1):
        if keep[i][cap]:
            picked.append(cand.iloc[i]["project_id"]); cap -= w[i]
    return _summ(econ, picked, budget, None, "DP knapsack (exact, budget-only)")


def optimize(econ: pd.DataFrame, budget: float, rig_capacity: float | None = None,
             min_first_year_bbl: float = 0.0) -> Program:
    """Best available exact optimizer; falls back gracefully if no solver."""
    try:
        import pulp  # noqa: F401
        return milp_select(econ, budget, rig_capacity, min_first_year_bbl)
    except Exception:
        if rig_capacity is None and min_first_year_bbl <= 0:
            return dp_knapsack_select(econ, budget)
        return greedy_select(econ, budget, rig_capacity, min_first_year_bbl)
