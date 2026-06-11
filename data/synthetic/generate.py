"""Generate a synthetic Permian capital backlog (~45 projects) — the inventory the
optimizer chooses from.

Design goals
------------
1. **Defensible, not flattering.** Type-curve, capex, opex, Pc and rig-day ranges are
   order-of-magnitude consistent with public Permian (Delaware/Midland) figures, so the
   resulting economics look like a real committee deck — a spread from strong wells down
   through marginal and genuinely *sub-economic* projects — not a backlog where everything
   prints money.
2. **Make the optimizer earn its keep.** Total backlog capex (~$170MM) and rig-days far
   exceed any single-year budget, and a deliberate tail of low-return / negative projects
   sits next to the good ones. That is what makes the budget + rig-day constraints *bind*
   and the MILP-vs-greedy gap a real result rather than an artifact of "pick everything."
3. **Lumpy capex** (multi-$MM drills next to sub-$MM workovers) so the 0/1 knapsack has
   genuine combinatorial structure for branch-and-bound to exploit over rank-and-cut.

Parameter ranges (order-of-magnitude, public sources)
-----------------------------------------------------
* **qi (peak oil bopd):** Permian horizontals commonly peak ~500-1,500 bo/d (EIA *Drilling
  Productivity Report*; operator investor decks). DUC completions sit in the same band
  (the wellbore is the same; only drilling is sunk). Recompletions/refracs add a fraction
  of a new-well rate (~150-400 bo/d); workovers/ESP jobs are small (~20-120 bo/d adds).
* **di (Arps nominal annual decline):** unconventional wells decline ~65-80% in year one,
  i.e. a *high* nominal annual decline ~1.3-2.1/yr (EIA DPR legacy-decline; SPE type-curve
  literature). Conventional workover adds decline slower (~1.0-1.8/yr).
* **b (hyperbolic exponent):** shale b typically ~0.8-1.1 (Arps 1945 bounded to b<=~1.1;
  values >1 over-estimate late-life rate and EUR — avoided here). Lower b (~0.3-0.7) for
  the more exponential workover/recompletion adds.
* **EUR (emergent):** the qi/di/b above yield ~250-850 Mbbl oil for new drills/DUCs and
  far less for the small jobs — consistent with single-well Permian oil EURs (~0.3-0.8
  MMbbl typical), NOT the ~1.0-1.5 MMbbl the prior fantasy curves produced.
* **capex (D&C, $):** new drill ~$7-10MM; DUC completion ~$3.5-6MM (drilling already
  spent); recompletion/refrac ~$1.5-3.5MM; workover ~$0.15-0.6MM (operator AFEs / EIA
  well-cost studies).
* **opex ($/bbl):** ~$8-16 LOE depending on lift/water cut.
* **Pc (chance of success):** lower for value-add recompletions/refracs (~0.55-0.78,
  real geologic/operational risk) than for drilling out a permitted DUC (~0.86-0.94).
* **nri:** ~0.75-0.82 (working-interest owner's share net of royalty).
* **rig_days:** rig/crew days consumed — drills 18-30, DUC completions 5-12, smaller jobs
  2-8 (these set the rig-fleet constraint).

Prices/discount are applied downstream in ``economics.py`` (effective-annual 10%). Seeded
for determinism so CI and the demo see the same inventory every run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
SEED = 7
AREAS = ["Delaware-A", "Delaware-B", "Midland-N", "Midland-S"]

# Each category is split into a "core" tranche (decent-to-strong economics) and a "tail"
# tranche (deliberately marginal / sub-economic: weaker qi, faster decline, higher capex
# or opex, lower Pc). The tail is what makes the budget/rig constraints actually bind.
#   spec tuple: (count, capex range, qi range, di range, b range, opex range, pc range, rig_days range)
SPEC = {
    # ---- core tranche: the projects a committee would want to fund ----
    # Rig-days are deliberately *heterogeneous* across categories: a new drill burns 3-5
    # weeks of rig/crew time while a DUC completion or workover is rig-light. That mismatch
    # between value-density (risked NPV / $) and rig intensity is what lets the rig-aware
    # MILP beat a greedy that ranks on capital efficiency alone (it over-spends rig time on
    # a few big drills); see the optimizer-vs-greedy gap in the evals.
    "new_drill":        (8, (7.5e6, 9.5e6), (700, 1200), (1.35, 1.70), (0.85, 1.05), (9, 13),  (0.84, 0.92), (22, 34)),
    "duc_completion":   (7, (3.8e6, 5.6e6), (650, 1050), (1.45, 1.80), (0.82, 1.00), (8, 12),  (0.88, 0.94), (5, 11)),
    "recompletion":     (6, (1.8e6, 3.2e6), (220, 400),  (1.50, 1.85), (0.55, 0.75), (10, 14), (0.62, 0.78), (4, 9)),
    "workover":         (8, (0.18e6, 0.45e6), (55, 120), (1.10, 1.55), (0.35, 0.55), (11, 15), (0.82, 0.92), (2, 5)),
    "alift_conversion": (3, (0.35e6, 0.7e6), (60, 130),  (1.05, 1.45), (0.40, 0.65), (10, 14), (0.78, 0.90), (3, 6)),
    # ---- marginal / sub-economic tail: low return or negative risked NPV at the base deck ----
    "new_drill_tail":   (3, (9.0e6, 10.5e6), (430, 600), (1.85, 2.15), (0.78, 0.92), (12, 16), (0.70, 0.80), (26, 38)),
    "duc_completion_tail": (2, (5.0e6, 6.2e6), (430, 560), (1.80, 2.05), (0.78, 0.90), (11, 15), (0.80, 0.88), (8, 12)),
    "recompletion_tail": (4, (2.8e6, 3.8e6), (110, 200), (1.85, 2.20), (0.45, 0.62), (13, 17), (0.52, 0.66), (5, 10)),
    "workover_tail":    (4, (0.45e6, 0.7e6), (20, 45),   (1.55, 2.00), (0.28, 0.42), (14, 18), (0.74, 0.86), (3, 6)),
}

# tail categories are reported under their base category label (a "new_drill_tail" is still
# a new drill — just a poor one); this maps the generator's tranche key to the real category.
BASE_CATEGORY = {
    "new_drill_tail": "new_drill",
    "duc_completion_tail": "duc_completion",
    "recompletion_tail": "recompletion",
    "workover_tail": "workover",
}


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    n = 0
    for cat, (count, capex_r, qi_r, di_r, b_r, opex_r, pc_r, rig_r) in SPEC.items():
        category = BASE_CATEGORY.get(cat, cat)
        for _ in range(count):
            n += 1
            rows.append({
                "project_id": f"P{n:03d}",
                "name": f"{category.split('_')[0].title()}-{n:03d}",
                "category": category,
                "area": AREAS[rng.integers(0, len(AREAS))],
                "capex_usd": round(rng.uniform(*capex_r), -3),
                "qi_bopd": round(rng.uniform(*qi_r), 1),
                "di_annual": round(rng.uniform(*di_r), 3),
                "b": round(rng.uniform(*b_r), 2),
                "opex_per_bbl": round(rng.uniform(*opex_r), 2),
                "nri": round(rng.uniform(0.75, 0.82), 4),
                "pc": round(rng.uniform(*pc_r), 3),
                "rig_days": int(rng.integers(rig_r[0], rig_r[1] + 1)),
                "earliest_quarter": int(rng.integers(1, 5)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "projects.csv", index=False)
    print(f"Wrote {len(df)} projects to {OUT / 'projects.csv'} "
          f"(total capex ${df['capex_usd'].sum()/1e6:,.0f}MM, total rig-days {df['rig_days'].sum()}).")


if __name__ == "__main__":
    main()
