"""Capital-program memo writer. The optimization is deterministic; the LLM only writes
the recommendation memo a VP / capital committee would read. Deterministic fallback so
it runs with no key (bring-your-own-key)."""
from __future__ import annotations

import json
import os
from datetime import date

from dotenv import load_dotenv


class MissingAPIKey(RuntimeError):
    pass


SYSTEM_PROMPT = """You are a Staff Production Engineer presenting the recommended capital program to the asset VP / capital committee. You are given a deterministic optimization result (the program that maximizes risked NPV under the budget + rig constraints), the greedy baseline it beat, the by-category split, and price sensitivity. Write a one-page markdown memo:

1. **# {year} Capital Program — Recommendation**
2. **## Recommendation** — 2-3 sentences: the program's risked NPV, capex deployed vs. budget, production add (first-year bbl), and weighted capital efficiency. Lead with the value.
3. **## Why this program** — the allocation by category (drill / DUC / recompletion / workover / conversion) and the logic; note the rig-day utilization.
4. **## The optimization is worth it** — state the $ of risked NPV captured OVER the naive rank-by-return-and-cut baseline, and that the solution is provably within the stated optimality gap of the bound.
5. **## Price risk** — how the program / NPV holds at the low price case vs. base.
6. **## The ask** — approve $X of capital for N projects; name the 1-2 highest-value projects.

Use the numbers verbatim. Terse, decision-ready. First character must be '#'."""


def render_memo_markdown(program, greedy, price_df, by_cat: dict, year: int = 2026) -> str:
    L = [f"# {year} Capital Program — Recommendation", "", "## Recommendation"]
    uplift = program.risked_npv - greedy.risked_npv
    L.append(
        f"Deploy **${program.capex_used/1e6:,.1f}MM** of the **${program.budget/1e6:,.0f}MM** budget across "
        f"**{program.n_selected} projects** for **${program.risked_npv/1e6:,.1f}MM risked NPV** "
        f"(+{program.first_year_bbl:,.0f} bbl first-year, capital efficiency "
        f"**{program.weighted_cap_eff:.2f}x**).")
    L += ["", "## Allocation by category", "", "| Category | Projects | Capex | Risked NPV |", "|---|---|---|---|"]
    for cat, v in sorted(by_cat.items(), key=lambda kv: -kv[1]["risked_npv"]):
        L.append(f"| {cat} | {v['n']} | ${v['capex']/1e6:,.1f}MM | ${v['risked_npv']/1e6:,.1f}MM |")
    L += ["", "## The optimization is worth it",
          f"The optimizer captures **${uplift/1e6:,.1f}MM** more risked NPV than rank-by-return-and-cut "
          f"at the same budget"
          + (f", and is within **{program.optimality_gap_pct:.1f}%** of the LP bound (provably near-optimal)."
             if program.optimality_gap_pct is not None else ".")]
    if price_df is not None and len(price_df):
        lo = price_df.iloc[0]; base = price_df[price_df["price"] == 70.0]
        base = base.iloc[0] if len(base) else price_df.iloc[len(price_df)//2]
        L += ["", "## Price risk",
              f"At ${lo['price']:.0f}/bbl the optimal program is **${lo['risked_npv']/1e6:,.1f}MM** risked NPV "
              f"({lo['n_selected']:.0f} projects) vs **${base['risked_npv']/1e6:,.1f}MM** at ${base['price']:.0f}."]
    L += ["", "## The ask",
          f"Approve **${program.capex_used/1e6:,.1f}MM** for **{program.n_selected}** projects."]
    return "\n".join(L)


def write_memo(program, greedy, price_df, by_cat, year: int = 2026,
               model: str | None = None, client=None) -> str:
    load_dotenv()
    if client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKey("No ANTHROPIC_API_KEY — use render_memo_markdown() for the deterministic memo.")
        from anthropic import Anthropic
        client = Anthropic(api_key=key)
    model = model or os.environ.get("MODEL", "claude-sonnet-4-6")
    payload = {
        "year": year,
        "program": {k: getattr(program, k) for k in
                    ("risked_npv", "npv", "capex_used", "budget", "rig_used", "rig_capacity",
                     "n_selected", "first_year_bbl", "weighted_cap_eff", "optimality_gap_pct")},
        "greedy_risked_npv": greedy.risked_npv,
        "by_category": by_cat,
        "price_sensitivity": price_df.to_dict("records") if price_df is not None else [],
    }
    user = f"Date: {date.today().isoformat()}\n\nOptimization result:\n{json.dumps(payload, indent=2, default=str)}\n\nWrite the memo."
    resp = client.messages.create(model=model, max_tokens=1600, system=SYSTEM_PROMPT.format(year=year),
                                  messages=[{"role": "user", "content": user}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    h = text.find("\n#")
    if h > 0 and not text.lstrip().startswith("#"):
        text = text[h:].lstrip()
    return text
