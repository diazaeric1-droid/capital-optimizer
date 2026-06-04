"""Tests for per-project economics."""
from src.economics import project_economics
from src.projects import Project


def _proj(**kw):
    base = dict(project_id="P1", name="n", category="new_drill", area="A", capex_usd=7e6,
                qi_bopd=800, di_annual=0.7, b=1.1, opex_per_bbl=10, nri=0.79, pc=0.9,
                rig_days=24, earliest_quarter=1)
    base.update(kw)
    return Project(**base)


def test_good_drill_has_positive_npv_and_sane_metrics():
    e = project_economics(_proj(), realized_price=70.0)
    assert e.npv_usd > 0 and e.risked_npv_usd > 0
    assert e.eur_bbl > e.first_year_bbl > 0
    assert e.payout_months > 0
    assert e.capital_efficiency > 0
    assert e.fd_per_bbl > 0


def test_risked_npv_is_below_unrisked():
    e = project_economics(_proj(pc=0.7), realized_price=70.0)
    assert e.risked_npv_usd < e.npv_usd          # risk discount + dry-hole downside


def test_marginal_project_can_be_negative_risked():
    # tiny rate, big capex, low Pc -> should be value-destructive
    e = project_economics(_proj(capex_usd=9e6, qi_bopd=80, pc=0.6), realized_price=45.0)
    assert e.risked_npv_usd < 0


def test_higher_price_lifts_npv():
    lo = project_economics(_proj(), 50.0).npv_usd
    hi = project_economics(_proj(), 80.0).npv_usd
    assert hi > lo
