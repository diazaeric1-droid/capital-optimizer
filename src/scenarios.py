"""Scenario analysis — how the optimal program shifts with the price deck and the
budget level (the questions a VP always asks: "what if oil is $55?" and "what's the
next $10MM of capital actually worth?")."""
from __future__ import annotations

import pandas as pd

from .economics import economics_frame
from .optimizer import optimize
from .projects import Project


def price_scenarios(projects: list[Project], budget: float, rig_capacity: float | None,
                    prices=(50.0, 60.0, 70.0, 80.0), min_first_year_bbl: float = 0.0) -> pd.DataFrame:
    rows = []
    for px in prices:
        econ = economics_frame(projects, px)
        prog = optimize(econ, budget, rig_capacity, min_first_year_bbl)
        rows.append({"price": px, "n_selected": prog.n_selected, "capex_used": prog.capex_used,
                     "risked_npv": prog.risked_npv, "first_year_bbl": prog.first_year_bbl})
    return pd.DataFrame(rows)


def budget_frontier(projects: list[Project], price: float, max_budget: float,
                    rig_capacity: float | None, steps: int = 16) -> pd.DataFrame:
    """Efficient frontier: optimal risked NPV as a function of the capital budget.
    The curve flattens — the marginal value of capital diminishes — which is exactly
    the picture that justifies (or caps) the budget ask."""
    econ = economics_frame(projects, price)
    rows = []
    for i in range(1, steps + 1):
        b = max_budget * i / steps
        prog = optimize(econ, b, rig_capacity)
        rows.append({"budget": b, "risked_npv": prog.risked_npv,
                     "capex_used": prog.capex_used, "n_selected": prog.n_selected})
    return pd.DataFrame(rows)
