"""Per-project economics — the deterministic value each capital project creates.

Arps type curve → monthly volumes → discounted cash flow. We report the numbers a
capital committee ranks on: NPV, risked NPV (chance-of-success weighted), IRR, payout,
EUR, F&D ($/bbl), and capital efficiency (discounted NPV per $ capex). Effective-annual
discounting (a 10% input means 10%/yr, not 10.47% from monthly compounding).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import econ_core
from .projects import Project

DEFAULT_DISCOUNT = 0.10
DEFAULT_HORIZON_YEARS = 15
DAYS_PER_MONTH = econ_core.DAYS_PER_MONTH


@dataclass
class ProjectEconomics:
    project_id: str
    capex_usd: float
    npv_usd: float                # success-case NPV (PV of net revenue − capex)
    risked_npv_usd: float         # Pc·NPV_success + (1−Pc)·(−capex)   [dry-hole downside]
    irr_pct: float | None
    payout_months: float
    eur_bbl: float
    first_year_bbl: float
    fd_per_bbl: float             # capex / EUR
    capital_efficiency: float     # risked NPV / capex  (the ranking metric)
    pc: float
    rig_days: float


def project_economics(p: Project, realized_price: float, discount: float = DEFAULT_DISCOUNT,
                      horizon_years: int = DEFAULT_HORIZON_YEARS) -> ProjectEconomics:
    months = np.arange(1, horizon_years * 12 + 1)
    rate = econ_core.arps_monthly_rate(p.qi_bopd, p.di_annual, p.b, months)
    vol = np.maximum(rate, 0.0) * DAYS_PER_MONTH                    # bbl/month
    margin = (realized_price - p.opex_per_bbl) * p.nri              # $/bbl net to operator
    monthly_cf = vol * margin

    pv = econ_core.discounted_pv(monthly_cf, discount)
    npv = pv - p.capex_usd
    # Risked NPV = pc·PV − capex (cost is certain; only revenue is chance-weighted). This is
    # algebraically identical to the dry-hole framing pc·NPV_success + (1−pc)·(−capex).
    risked = econ_core.risked_npv(pv, p.capex_usd, p.pc)

    payout = econ_core.payout_months(monthly_cf, p.capex_usd)

    eur = float(vol.sum())
    fy = float(vol[:12].sum())
    fd = p.capex_usd / eur if eur > 0 else float("inf")
    cap_eff = risked / p.capex_usd if p.capex_usd > 0 else 0.0

    return ProjectEconomics(
        project_id=p.project_id, capex_usd=p.capex_usd, npv_usd=npv, risked_npv_usd=risked,
        irr_pct=econ_core.irr_annual(monthly_cf, p.capex_usd), payout_months=payout,
        eur_bbl=eur, first_year_bbl=fy, fd_per_bbl=fd, capital_efficiency=cap_eff,
        pc=p.pc, rig_days=p.rig_days,
    )


def economics_frame(projects: list[Project], realized_price: float,
                    discount: float = DEFAULT_DISCOUNT) -> "object":
    import pandas as pd
    rows = []
    for p in projects:
        e = project_economics(p, realized_price, discount)
        rows.append({"project_id": p.project_id, "name": p.name, "category": p.category,
                     "label": p.label, "area": p.area, **e.__dict__})
    df = pd.DataFrame(rows)
    return df.loc[:, ~df.columns.duplicated()]
