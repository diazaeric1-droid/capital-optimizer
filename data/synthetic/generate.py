"""Generate a synthetic capital backlog: ~45 projects across categories with type-curve
parameters, capex, chance of success, and rig-days — the inventory the optimizer chooses
from. Capex is deliberately *lumpy* (big drills vs. small workovers) so the knapsack has
real structure and the MILP can beat the greedy rank-and-cut.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
SEED = 7
AREAS = ["Delaware-A", "Delaware-B", "Midland-N", "Midland-S"]

# category -> (count, capex range, qi range, di range, b range, opex range, pc range, rig_days range)
SPEC = {
    "new_drill":       (10, (6.0e6, 9.0e6), (550, 1150), (0.60, 0.85), (0.9, 1.3), (8, 12), (0.83, 0.93), (18, 30)),
    "duc_completion":  (10, (3.0e6, 5.0e6), (450, 950),  (0.60, 0.85), (1.0, 1.3), (8, 12), (0.86, 0.94), (6, 12)),
    "recompletion":    (10, (0.8e6, 2.0e6), (120, 380),  (0.45, 0.70), (0.6, 1.0), (10, 14), (0.62, 0.80), (4, 8)),
    "workover":        (10, (0.1e6, 0.5e6), (25, 120),   (0.30, 0.60), (0.3, 0.7), (12, 16), (0.80, 0.93), (2, 5)),
    "alift_conversion": (5, (0.3e6, 0.8e6), (40, 150),   (0.30, 0.55), (0.4, 0.8), (10, 14), (0.76, 0.88), (3, 6)),
}


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    n = 0
    for cat, (count, capex_r, qi_r, di_r, b_r, opex_r, pc_r, rig_r) in SPEC.items():
        for _ in range(count):
            n += 1
            rows.append({
                "project_id": f"P{n:03d}",
                "name": f"{cat.split('_')[0].title()}-{n:03d}",
                "category": cat,
                "area": AREAS[rng.integers(0, len(AREAS))],
                "capex_usd": round(rng.uniform(*capex_r), -3),
                "qi_bopd": round(rng.uniform(*qi_r), 1),
                "di_annual": round(rng.uniform(*di_r), 3),
                "b": round(rng.uniform(*b_r), 2),
                "opex_per_bbl": round(rng.uniform(*opex_r), 2),
                "nri": round(rng.uniform(0.78, 0.80), 4),
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
