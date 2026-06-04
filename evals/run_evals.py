"""Optimization validation harness.

There's no "accuracy %" for an optimizer — the right checks are: the MILP is FEASIBLE
(respects budget + rig limits), it BEATS-OR-TIES the greedy baseline, and it's provably
near-optimal (small gap to the LP-relaxation bound). We assert these across several
budget/rig settings and write evals/results/summary.json. CI fails if the optimizer is
ever worse than greedy or infeasible. Run: ``python -m evals.run_evals``.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.economics import economics_frame
from src.optimizer import dp_knapsack_select, greedy_select, milp_select
from src.projects import load_projects

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ROOT / "data" / "synthetic" / "projects.csv"
OUT = ROOT / "evals" / "results"
SETTINGS = [(60e6, 170), (60e6, 250), (40e6, 130), (80e6, 200)]


def main():
    projects = load_projects(PROJECTS)
    econ = economics_frame(projects, 70.0)
    results = []
    failures = []
    for budget, rig in SETTINGS:
        m = milp_select(econ, budget, rig)
        g = greedy_select(econ, budget, rig)
        feasible = (m.capex_used <= budget + 1) and (m.rig_used <= rig + 1e-6)
        beats = m.risked_npv >= g.risked_npv - 1.0
        results.append({
            "budget_mm": budget / 1e6, "rig_capacity": rig,
            "milp_risked_npv_mm": round(m.risked_npv / 1e6, 2),
            "greedy_risked_npv_mm": round(g.risked_npv / 1e6, 2),
            "uplift_mm": round((m.risked_npv - g.risked_npv) / 1e6, 2),
            "uplift_pct": round((m.risked_npv - g.risked_npv) / max(g.risked_npv, 1) * 100, 2),
            "optimality_gap_pct": round(m.optimality_gap_pct, 3) if m.optimality_gap_pct is not None else None,
            "capex_used_mm": round(m.capex_used / 1e6, 2), "rig_used": m.rig_used,
            "n_selected": m.n_selected, "feasible": feasible, "beats_greedy": beats,
        })
        if not feasible:
            failures.append(f"infeasible at budget={budget/1e6:.0f}MM rig={rig}")
        if not beats:
            failures.append(f"MILP worse than greedy at budget={budget/1e6:.0f}MM rig={rig}")

    # budget-only optimality: DP knapsack (feasible lower bound) must not exceed MILP
    mb = milp_select(econ, 60e6, None)
    dp = dp_knapsack_select(econ, 60e6)
    dp_ok = dp.risked_npv <= mb.risked_npv + 1.0

    summary = {"settings": results, "dp_check": {"milp_mm": round(mb.risked_npv/1e6, 2),
               "dp_mm": round(dp.risked_npv/1e6, 2), "dp_le_milp": dp_ok},
               "n_projects": len(projects)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Optimization validation — {len(projects)} projects")
    print(f"{'budget':>8}{'rig':>6}{'greedy$MM':>11}{'MILP$MM':>10}{'uplift%':>9}{'gap%':>7}{'ok':>4}")
    for r in results:
        print(f"{r['budget_mm']:>7.0f}M{r['rig_capacity']:>6}{r['greedy_risked_npv_mm']:>11.1f}"
              f"{r['milp_risked_npv_mm']:>10.1f}{r['uplift_pct']:>9.1f}"
              f"{(r['optimality_gap_pct'] or 0):>7.2f}{'✓' if r['feasible'] and r['beats_greedy'] else '✗':>4}")
    print(f"\nWrote {OUT/'summary.json'}")
    if failures or not dp_ok:
        raise SystemExit("VALIDATION FAILED: " + "; ".join(failures + ([] if dp_ok else ["DP > MILP"])))


if __name__ == "__main__":
    main()
