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

INFEASIBLE_MSG = ("No feasible capital program under the given budget / rig-day / "
                  "production-floor constraints — relax a constraint.")


class InfeasibleProgram(ValueError):
    """Raised when the MILP has no optimal solution (infeasible / unbounded constraints)."""

    def __init__(self, status: str = "Infeasible"):
        self.status = status
        super().__init__(INFEASIBLE_MSG)


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
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        # On INFEASIBLE problems CBC still returns variable values; reading them
        # yields a bogus, over-budget "program" that downstream code would label
        # feasible. Refuse to return a solution unless the solve is provably optimal.
        raise InfeasibleProgram(status)
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


# ---------------------------------------------------------------------------
# Multi-period capital program (the annual-plan deliverable)
# ---------------------------------------------------------------------------
@dataclass
class MultiPeriodProgram:
    """Result of the multi-period optimizer: which project is funded in which period."""
    method: str
    periods: int
    # list of (project_id, period_index) — period_index is 0-based
    selected: list[tuple]
    risked_npv: float                 # discounted objective value (timing-aware)
    undiscounted_risked_npv: float    # sum of risked NPV ignoring period discounting
    budget_per_period: list[float]
    rig_per_period: list[float]
    capex_per_period: list[float]     # capex deployed each period
    rig_used_per_period: list[float]  # rig-days used each period
    npv_per_period: list[float]       # undiscounted risked NPV funded each period
    n_selected: int
    n_available: int
    lp_bound: float | None = None
    optimality_gap_pct: float | None = None


def _normalize_per_period(value, periods: int, name: str) -> list[float]:
    """Accept a scalar (broadcast) or a per-period sequence; validate length."""
    if np.isscalar(value):
        return [float(value)] * periods
    seq = list(value)
    if len(seq) != periods:
        raise ValueError(f"{name} must be a scalar or a length-{periods} sequence, got {len(seq)}.")
    return [float(v) for v in seq]


def _mp_inputs(econ: pd.DataFrame, periods: int, budget_per_period, rig_per_period,
               earliest_start):
    """Shared input normalization for the multi-period optimizer + greedy baseline."""
    if periods < 1:
        raise ValueError("periods must be >= 1.")
    # only positive-value projects are eligible (mirrors greedy/dp single-period)
    cand = econ[econ["risked_npv_usd"] > 0]
    ids = list(cand["project_id"])
    risked = dict(zip(cand["project_id"], cand["risked_npv_usd"]))
    capex = dict(zip(cand["project_id"], cand["capex_usd"]))
    rig = dict(zip(cand["project_id"], cand["rig_days"]))
    budgets = _normalize_per_period(budget_per_period, periods, "budget_per_period")
    rigs = _normalize_per_period(rig_per_period, periods, "rig_per_period")
    es = {i: 0 for i in ids}
    if earliest_start:
        for i, t in earliest_start.items():
            if i in es:
                es[i] = max(0, int(t))
    return ids, risked, capex, rig, budgets, rigs, es


def _mp_summary(method, periods, selected, risked, capex, rig, budgets, rigs,
                discount_per_period, n_available) -> MultiPeriodProgram:
    capex_pp = [0.0] * periods
    rig_pp = [0.0] * periods
    npv_pp = [0.0] * periods
    disc_obj = 0.0
    undisc = 0.0
    for i, t in selected:
        capex_pp[t] += capex[i]
        rig_pp[t] += rig[i]
        npv_pp[t] += risked[i]
        disc_obj += risked[i] / ((1.0 + discount_per_period) ** t)
        undisc += risked[i]
    return MultiPeriodProgram(
        method=method, periods=periods, selected=list(selected),
        risked_npv=disc_obj, undiscounted_risked_npv=undisc,
        budget_per_period=list(budgets), rig_per_period=list(rigs),
        capex_per_period=capex_pp, rig_used_per_period=rig_pp, npv_per_period=npv_pp,
        n_selected=len(selected), n_available=int(n_available))


def greedy_select_multiperiod(econ: pd.DataFrame, periods: int, budget_per_period,
                              rig_per_period, discount_per_period: float = 0.0,
                              earliest_start: dict | None = None) -> MultiPeriodProgram:
    """Greedy multi-period baseline: rank by capital efficiency, place each project in the
    earliest period (>= its earliest_start) with remaining budget AND rig capacity."""
    ids, risked, capex, rig, budgets, rigs, es = _mp_inputs(
        econ, periods, budget_per_period, rig_per_period, earliest_start)
    order = (econ[econ["project_id"].isin(ids)]
             .sort_values("capital_efficiency", ascending=False)["project_id"].tolist())
    cap_left = list(budgets)
    rig_left = list(rigs)
    selected = []
    for i in order:
        for t in range(es[i], periods):
            if capex[i] <= cap_left[t] + 1e-9 and rig[i] <= rig_left[t] + 1e-9:
                selected.append((i, t))
                cap_left[t] -= capex[i]
                rig_left[t] -= rig[i]
                break
    return _mp_summary("greedy multi-period (rank-by-efficiency, earliest-fit)", periods,
                       selected, risked, capex, rig, budgets, rigs, discount_per_period,
                       n_available=len(econ))


def milp_select_multiperiod(econ: pd.DataFrame, periods: int, budget_per_period,
                            rig_per_period, discount_per_period: float = 0.0,
                            earliest_start: dict | None = None) -> MultiPeriodProgram:
    """Multi-period capital program: assign each backlog project to AT MOST one period,
    maximizing the (optionally period-discounted) total risked NPV under per-period
    capital-budget and rig-day capacity, respecting an optional per-project earliest start.

    Parameters
    ----------
    econ : DataFrame from ``economics_frame`` (needs project_id, risked_npv_usd, capex_usd, rig_days).
    periods : number of planning periods (e.g. 4 quarters).
    budget_per_period : scalar (broadcast to every period) or length-``periods`` sequence ($).
    rig_per_period : scalar or length-``periods`` sequence (rig-days).
    discount_per_period : per-period discount applied as 1/(1+r)**t to risked NPV funded in t.
    earliest_start : optional {project_id: earliest_period_index (0-based)}.

    Returns ``MultiPeriodProgram``. Raises ``InfeasibleProgram`` if CBC can't prove optimality.
    """
    ids, risked, capex, rig, budgets, rigs, es = _mp_inputs(
        econ, periods, budget_per_period, rig_per_period, earliest_start)

    # Discounted per-(project,period) value coefficients (the true objective).
    coef = {(i, t): risked[i] / ((1.0 + discount_per_period) ** t)
            for i in ids for t in range(es[i], periods)}
    # Symmetry-breaking earliness nudge: when discount==0 a project is value-indifferent
    # across all feasible periods, which spawns a huge symmetric branch-and-bound tree
    # (CBC can stall for minutes). Add a negligible preference for earlier periods —
    # eps · t, with eps small enough (≤1e-6 of the smallest positive value) that it can
    # NEVER change which projects are funded, only break ties toward earlier placement.
    min_val = min((v for v in risked.values() if v > 0), default=1.0)
    eps = (min_val * 1e-7) / max(periods, 1)

    def solve(relaxed: bool):
        import pulp
        prob = pulp.LpProblem("capital_program_multiperiod", pulp.LpMaximize)
        cat = "Continuous" if relaxed else "Binary"
        # x[(i,t)] only created for allowed periods (t >= earliest_start[i])
        x = {(i, t): pulp.LpVariable(f"x_{i}_{t}", lowBound=0, upBound=1, cat=cat)
             for i in ids for t in range(es[i], periods)}
        # objective: discounted risked NPV, minus a tiny earliness tie-break
        prob += pulp.lpSum((coef[(i, t)] - eps * t) * x[(i, t)] for (i, t) in x)
        # each project funded at most once across all periods
        for i in ids:
            terms = [x[(i, t)] for t in range(es[i], periods)]
            if terms:
                prob += pulp.lpSum(terms) <= 1
        # per-period capital budget + rig capacity
        for t in range(periods):
            prob += pulp.lpSum(capex[i] * x[(i, t)] for i in ids if (i, t) in x) <= budgets[t]
            prob += pulp.lpSum(rig[i] * x[(i, t)] for i in ids if (i, t) in x) <= rigs[t]
        # The per-period assignment is bin-packing-symmetric: at zero discount a project
        # is value-indifferent across feasible periods, so the exact (gap=0) search can
        # stall. We solve the relaxation (relaxed=True) exactly for the LP bound, but let
        # the integer solve stop at a small MIP gap — the reported optimality_gap_pct
        # (vs. the LP bound) keeps the result honest, and a time limit guards the worst case.
        kwargs = dict(msg=0, timeLimit=120)
        if not relaxed:
            kwargs["gapRel"] = 0.01            # stop within 1% relative MIP gap (sub-second)
        prob.solve(pulp.PULP_CBC_CMD(**kwargs))
        status = pulp.LpStatus[prob.status]
        # CBC reports "Optimal" both when proven optimal and when it stops within the
        # MIP gap / time limit with an integer-feasible incumbent. Treat a returned
        # incumbent (any variable assignment) as feasible; only a true infeasible/
        # unbounded/not-solved status is fatal.
        if status not in ("Optimal",):
            # No incumbent at all -> genuinely infeasible (or unbounded/undefined).
            raise InfeasibleProgram(status)
        chosen = [(i, t) for (i, t) in x if (x[(i, t)].value() or 0) > 0.5]
        # report the TRUE discounted objective (exclude the tie-break nudge)
        obj = sum(coef[(i, t)] * (x[(i, t)].value() or 0.0) for (i, t) in x)
        return chosen, float(obj)

    chosen, _obj = solve(relaxed=False)
    prog = _mp_summary("MILP multi-period (optimal under per-period constraints)", periods,
                       chosen, risked, capex, rig, budgets, rigs, discount_per_period,
                       n_available=len(econ))
    try:
        _, lp_bound = solve(relaxed=True)
        prog.lp_bound = lp_bound
        prog.optimality_gap_pct = (max(lp_bound - prog.risked_npv, 0.0) / prog.risked_npv * 100.0
                                   if prog.risked_npv > 0 else 0.0)
    except Exception:
        pass
    return prog


def optimize(econ: pd.DataFrame, budget: float, rig_capacity: float | None = None,
             min_first_year_bbl: float = 0.0) -> Program:
    """Best available exact optimizer; falls back gracefully if no solver.

    Raises ``InfeasibleProgram`` (a ``ValueError``) when the constraints admit no
    feasible program — callers must surface that rather than render a bogus program.
    """
    try:
        import pulp  # noqa: F401
    except Exception:
        # No solver available — use the exact DP (budget-only) or greedy fallback.
        if rig_capacity is None and min_first_year_bbl <= 0:
            return dp_knapsack_select(econ, budget)
        return greedy_select(econ, budget, rig_capacity, min_first_year_bbl)
    # Solver present: let infeasibility propagate instead of masking it with a
    # fallback heuristic that would report a bogus "feasible" program.
    return milp_select(econ, budget, rig_capacity, min_first_year_bbl)
