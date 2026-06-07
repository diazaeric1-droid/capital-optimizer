# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/); SemVer.

## [0.2.1] — 2026-06-07
### Fixed
- **`ModuleNotFoundError: No module named 'theme'`** on deploy — `demo/app.py` now adds its own directory to `sys.path` so the vendored `theme` / `fleet_registry` resolve regardless of cwd (caught by CI render-smoke).
### Changed
- **Light theme** — suite-wide migration from dark/navy to a professional light palette (white surfaces, `plotly_white` charts, navy/blue accents retained); transparent fixed header so the title never clips. `runtime.txt` pinned to Python 3.11.

## [0.2.0] — 2026-06-06

### Added
- **Unified dark + navy suite theme** + a **cross-app sidebar suite navigator** for moving between the apps.
- **Multi-period capital MILP**: select which project runs in which period under per-period budget +
  rig-day capacity (fund-once, earliest-start, period-discounted NPV), with a **project×period
  schedule heatmap** + per-period utilization; **~12% / $35MM uplift over greedy** at ~1% optimality gap.
- **Funded-vs-rejected scatter** (capex × risked NPV, sized by rig-days).
- **Shared fleet registry**: Permian field/formation identity consistent across the suite.

### Fixed
- Infeasible programs no longer render a bogus "feasible ✓" plan (now a clear **infeasibility banner**).
- Empty-schedule `KeyError` crash.

### Changed
- Swept deprecated `use_container_width` (→ `width="stretch"`); requires `streamlit>=1.50`.

## [0.1.0] — 2026-06-04

Initial release — capital-allocation optimizer.

### Added
- **Per-project economics** (`src/economics.py`): Arps type curve → discounted cash flow → NPV, IRR
  (bisection), payout, EUR, F&D, **risked NPV** (Pc-weighted with dry-hole downside), and capital
  efficiency. Effective-annual discounting.
- **Optimizer** (`src/optimizer.py`): exact **MILP** (CBC via PuLP) maximizing risked NPV under a
  capital budget + rig-day capacity (+ optional production floor); a **greedy** rank-and-cut baseline;
  an exact **DP-knapsack** (budget-only) cross-check; and the **LP-relaxation bound** for an
  optimality gap. Graceful fallback to DP/greedy if no solver.
- **Scenarios** (`src/scenarios.py`): efficient frontier (NPV vs. budget) + price-deck sensitivity.
- **Scheduling** (`src/schedule.py`): lay the selected program into quarters under per-quarter rig
  capacity, respecting each project's earliest start.
- **Memo writer** (`src/narrator.py`): capital-committee recommendation — LLM-narrated (BYOK) with a
  deterministic fallback.
- **Validation harness** (`evals/run_evals.py`): asserts the MILP is feasible, beats-or-ties greedy,
  and is near-optimal across budget/rig settings; **CI gate**. On the synthetic backlog the optimizer
  captures ~$34MM (~12%) over rank-and-cut under a binding rig limit.
- **Streamlit app** + Docker/HF deploy + bring-your-own-key. Synthetic ~45-project backlog with lumpy
  capex so the knapsack has real structure.
