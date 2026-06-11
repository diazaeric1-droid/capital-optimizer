---
title: Capital Program Optimizer
emoji: 🛢️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: true
license: mit
---

# Capital Program Optimizer

> Ranks a drilling / DUC / workover backlog by **risked economics** and picks the program that
> **maximizes NPV under a capital budget + rig capacity** — the annual fight every asset VP and
> capital committee runs, turned into a solved optimization.

Built by a Staff Production Engineer (ex-OXY, ex-Shell) who sat in the capital-allocation meetings.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)

---

## The problem

Every year there are more projects — new drills, DUC completions, recompletions, workovers,
artificial-lift conversions — than there is capital or rigs to do them. The committee must pick the
*subset and sequence* that creates the most value under hard constraints. The default is "rank by
rate-of-return and cut at the budget line" — which **leaves money on the table** the moment a second
resource (rig crews) is also scarce, because a single-metric ranking can't trade off two constraints.

## What it does

1. **Risked project economics** (deterministic): an Arps type curve → discounted cash flow → **NPV,
   IRR, payout, F&D, EUR**, and **risked NPV** (chance-of-success weighted, with the dry-hole
   downside). Effective-annual discounting.
2. **Constrained optimization (MILP)**: selects the program that **maximizes total risked NPV**
   subject to the **capital budget** and **rig-day capacity** (and an optional production floor),
   solved exactly by branch-and-bound (CBC via PuLP).
3. **Proves it's worth it**: reports the $ of risked NPV captured over the greedy rank-and-cut
   baseline, plus the **LP-relaxation bound** so the solution is provably near-optimal (small gap).
   On the demo inventory under a binding rig limit, the optimizer captures **~$4–8MM (≈3–5%) more**
   than rank-and-cut.
4. **Scenarios a VP asks for**: the **efficient frontier** (optimal NPV vs. budget — diminishing
   returns), **price-deck sensitivity**, and a **quarterly schedule** under per-quarter rig capacity.
5. Deterministic optimization; the LLM only writes the capital-committee **memo**. Runs with no key.

## How it's validated

`python -m evals.run_evals` checks the MILP across several budget/rig settings: it must be
**feasible** (respect both limits), **beat-or-tie the greedy baseline**, and sit within a small
**optimality gap** of the LP bound; a DP-knapsack lower bound cross-checks the budget-only case.
**CI fails if the optimizer is ever worse than greedy or infeasible.**

## Quick start

```bash
pip install -e ".[demo,dev]"
python data/synthetic/generate.py     # ~45-project capital backlog
python -m evals.run_evals             # optimization validation
streamlit run demo/app.py
```

## The new capability this demonstrates

Beyond the rest of the suite, this adds **constrained optimization (MILP / knapsack) for capital
allocation** plus full single-project economics (NPV/IRR/payout/F&D/risked NPV) and the
efficient-frontier framing — the annual-plan deliverable that defines a production VP's year.

## License

MIT.

## Contact

Eric Diaz II — [LinkedIn](https://www.linkedin.com/in/eric-a-diaz2) — diaz.a.eric1@gmail.com
