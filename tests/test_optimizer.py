"""Tests for the optimizer: feasibility, optimality, and beats-or-ties the greedy baseline."""
import pandas as pd

from src.economics import economics_frame
from src.optimizer import dp_knapsack_select, greedy_select, milp_select
from src.projects import load_projects

PROJECTS = "data/synthetic/projects.csv"


def _econ():
    return economics_frame(load_projects(PROJECTS), 70.0)


def test_milp_is_feasible():
    econ = _econ()
    p = milp_select(econ, 60e6, 170)
    assert p.capex_used <= 60e6 + 1
    assert p.rig_used <= 170 + 1e-6


def test_milp_beats_or_ties_greedy_under_binding_rig():
    econ = _econ()
    m = milp_select(econ, 60e6, 170)
    g = greedy_select(econ, 60e6, 170)
    assert m.risked_npv >= g.risked_npv - 1.0
    assert m.risked_npv > g.risked_npv          # rig binds -> optimizer strictly better here


def test_optimality_gap_is_small():
    p = milp_select(_econ(), 60e6, 170)
    assert p.optimality_gap_pct is not None and p.optimality_gap_pct < 2.0


def test_dp_knapsack_is_feasible_lower_bound():
    econ = _econ()
    dp = dp_knapsack_select(econ, 60e6)
    mb = milp_select(econ, 60e6, None)
    # DP ceils weights -> always budget-feasible -> never exceeds the true (MILP) optimum
    assert dp.risked_npv <= mb.risked_npv + 1.0


def test_only_positive_projects_selected():
    econ = _econ()
    p = milp_select(econ, 60e6, 170)
    sel = econ[econ["project_id"].isin(p.selected_ids)]
    assert (sel["risked_npv_usd"] > 0).all()
