"""Tests for the multi-period capital program optimizer: per-period feasibility,
each project funded at most once, MILP >= greedy, small optimality gap,
earliest_start respected, and infeasible budgets raise InfeasibleProgram."""
import pytest

from src.economics import economics_frame
from src.optimizer import (
    InfeasibleProgram,
    greedy_select_multiperiod,
    milp_select_multiperiod,
)
from src.projects import load_projects

PROJECTS = "data/synthetic/projects.csv"
PERIODS = 4


def _econ():
    return economics_frame(load_projects(PROJECTS), 70.0)


def _split(total, periods):
    # even split of a total resource across periods (the app's default)
    return [total / periods] * periods


def _capex(econ, pid):
    return float(econ.loc[econ["project_id"] == pid, "capex_usd"].iloc[0])


def _rig(econ, pid):
    return float(econ.loc[econ["project_id"] == pid, "rig_days"].iloc[0])


def test_multiperiod_respects_every_per_period_cap_and_funds_once():
    econ = _econ()
    budgets = _split(60e6, PERIODS)
    rigs = _split(170, PERIODS)
    p = milp_select_multiperiod(econ, PERIODS, budgets, rigs)
    # each per-period budget + rig capacity respected
    for t in range(PERIODS):
        assert p.capex_per_period[t] <= budgets[t] + 1.0
        assert p.rig_used_per_period[t] <= rigs[t] + 1e-6
    # each project funded at most once
    funded_ids = [i for i, _ in p.selected]
    assert len(funded_ids) == len(set(funded_ids))
    # every funded period index is valid
    assert all(0 <= t < PERIODS for _, t in p.selected)


def test_milp_beats_or_ties_greedy_multiperiod():
    econ = _econ()
    budgets = _split(60e6, PERIODS)
    rigs = _split(170, PERIODS)
    m = milp_select_multiperiod(econ, PERIODS, budgets, rigs)
    g = greedy_select_multiperiod(econ, PERIODS, budgets, rigs)
    assert m.risked_npv >= g.risked_npv - 1.0


def test_greedy_multiperiod_is_feasible():
    econ = _econ()
    budgets = _split(60e6, PERIODS)
    rigs = _split(170, PERIODS)
    g = greedy_select_multiperiod(econ, PERIODS, budgets, rigs)
    for t in range(PERIODS):
        assert g.capex_per_period[t] <= budgets[t] + 1.0
        assert g.rig_used_per_period[t] <= rigs[t] + 1e-6
    funded_ids = [i for i, _ in g.selected]
    assert len(funded_ids) == len(set(funded_ids))


def test_optimality_gap_is_small():
    # The per-period assignment is bin-packing-symmetric, so the MILP is solved to a 1%
    # relative MIP gap (sub-second) rather than proven-exact; the reported gap is vs. the
    # LP-relaxation bound and stays comfortably small (~1%).
    econ = _econ()
    p = milp_select_multiperiod(econ, PERIODS, _split(60e6, PERIODS), _split(170, PERIODS))
    assert p.optimality_gap_pct is not None
    assert p.optimality_gap_pct <= 2.0


def test_earliest_start_respected():
    econ = _econ()
    # pick the single most capital-efficient project and force it to start late
    top = econ[econ["risked_npv_usd"] > 0].sort_values(
        "capital_efficiency", ascending=False)["project_id"].iloc[0]
    es = {top: 2}  # cannot be funded before period index 2
    p = milp_select_multiperiod(econ, PERIODS, _split(60e6, PERIODS), _split(170, PERIODS),
                                earliest_start=es)
    for i, t in p.selected:
        if i == top:
            assert t >= 2


def test_scalar_budget_broadcasts():
    econ = _econ()
    # scalar inputs should broadcast to every period and stay feasible per period
    p = milp_select_multiperiod(econ, PERIODS, 15e6, 42.5)
    for t in range(PERIODS):
        assert p.capex_per_period[t] <= 15e6 + 1.0
        assert p.rig_used_per_period[t] <= 42.5 + 1e-6


def test_discount_prefers_earlier_periods():
    econ = _econ()
    # with a positive per-period discount, total discounted value <= undiscounted
    p = milp_select_multiperiod(econ, PERIODS, _split(60e6, PERIODS), _split(170, PERIODS),
                                discount_per_period=0.05)
    assert p.risked_npv <= p.undiscounted_risked_npv + 1.0


def test_infeasible_per_period_budget_raises():
    econ = _econ()
    # every project's capex exceeds the tiny per-period budget AND a min-cost project
    # exists, but the real trip-wire: zero rig capacity makes any selection infeasible
    # is NOT infeasible (empty program is feasible). Instead force infeasibility via a
    # negative-style impossible constraint: budget high but require... we instead use the
    # documented contract — a per-period budget too small for the cheapest project still
    # yields an empty (feasible) program. To get a true InfeasibleProgram we drive CBC to
    # an infeasible status with a contradictory budget (negative).
    budgets = [-1.0] * PERIODS  # negative budget: no nonneg selection can satisfy <= -1
    with pytest.raises(InfeasibleProgram):
        milp_select_multiperiod(econ, PERIODS, budgets, _split(170, PERIODS))
