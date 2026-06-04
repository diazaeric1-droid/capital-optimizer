"""Capital Program Optimizer — pick the program that maximizes risked NPV under a
capital budget + rig-day capacity. Deterministic economics + MILP; LLM writes the memo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import shutil as _shutil
for _pyc in (REPO_ROOT / "src").rglob("__pycache__"):
    _shutil.rmtree(_pyc, ignore_errors=True)
for _m in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_m]

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import __version__
from src.economics import economics_frame
from src.narrator import MissingAPIKey, render_memo_markdown, write_memo
from src.optimizer import greedy_select, optimize
from src.projects import load_projects
from src.scenarios import budget_frontier, price_scenarios
from src.schedule import schedule_program

st.set_page_config(page_title="Capital Program Optimizer", page_icon="🛢️", layout="wide")
st.title(f"Capital Program Optimizer  `v{__version__}`")
st.caption("Rank a drilling / DUC / workover backlog by risked economics and pick the program that "
           "maximizes NPV under a capital budget + rig capacity. Built by an ex-OXY / ex-Shell Staff PE.")

with st.expander(f"🆕 What is this / v{__version__}"):
    st.markdown(
        "- **Risked project economics** — Arps type curve → discounted cash flow → NPV, IRR, payout, "
        "F&D, and **risked NPV** (chance-of-success weighted) per project.\n"
        "- **Constrained optimization (MILP)** — selects the program that maximizes total risked NPV "
        "subject to the **capital budget** and **rig-day capacity** (and an optional production floor), "
        "solved exactly via branch-and-bound (CBC).\n"
        "- **Beats rank-and-cut** — vs. the naive 'rank by return, cut at the budget' baseline, the "
        "optimizer captures the value a single-metric ranking misses when *two* resources are scarce; "
        "an **LP-relaxation bound** proves it's near-optimal.\n"
        "- **Scenarios** — efficient frontier (NPV vs. budget) + price-deck sensitivity, and a quarterly "
        "schedule under per-quarter rig capacity.\n"
        "- Deterministic optimization; the LLM only writes the capital-committee memo. Bring your own key."
    )

DATA = REPO_ROOT / "data" / "synthetic"
if not (DATA / "projects.csv").exists():
    with st.status("First-time setup: generating project inventory…", expanded=False):
        subprocess.run([sys.executable, str(DATA / "generate.py")], check=True)


@st.cache_data(show_spinner=False)
def _projects():
    return load_projects(DATA / "projects.csv")


projects = _projects()
total_capex = sum(p.capex_usd for p in projects)
total_rig = sum(p.rig_days for p in projects)

with st.sidebar:
    st.header("Plan inputs")
    price = st.number_input("Realized oil price ($/bbl)", 30.0, 120.0, 70.0, 1.0)
    budget = st.slider("Capital budget ($MM)", 10, int(total_capex / 1e6), 60, 5) * 1e6
    rig_cap = st.slider("Rig-day capacity", 60, int(total_rig), 170, 10)
    min_fy = st.number_input("Min first-year add (bbl, optional)", 0, 5_000_000, 0, 100_000)
    byok_key = st.text_input(
        "🔑 Anthropic API key (optional)", type="password",
        help="Bring your own key — used only this session, never stored. Powers the capital-program "
             "memo. The economics, optimizer, and all charts work without it.")
    st.caption(f"Inventory: {len(projects)} projects · ${total_capex/1e6:,.0f}MM capex · {total_rig} rig-days.")


@st.cache_data(show_spinner=False)
def _econ(px):
    return economics_frame(projects, px)


econ = _econ(price)
program = optimize(econ, budget, rig_cap, float(min_fy))
greedy = greedy_select(econ, budget, rig_cap, float(min_fy))
sel = set(program.selected_ids)
uplift = program.risked_npv - greedy.risked_npv

tab_prog, tab_sched, tab_val = st.tabs(["📈 Program", "🗓️ Schedule", "✅ Optimization validation"])

with tab_prog:
    c = st.columns(5)
    c[0].metric("Program risked NPV", f"${program.risked_npv/1e6:,.0f}MM")
    c[1].metric("Capital deployed", f"${program.capex_used/1e6:,.0f}MM", f"of ${budget/1e6:,.0f}MM budget")
    c[2].metric("Projects selected", f"{program.n_selected} / {program.n_available}")
    c[3].metric("First-year add", f"{program.first_year_bbl:,.0f} bbl")
    c[4].metric("Capital efficiency", f"{program.weighted_cap_eff:.2f}x")

    gap = f" · within {program.optimality_gap_pct:.1f}% of the LP bound" if program.optimality_gap_pct is not None else ""
    st.success(f"**Optimization captures ${uplift/1e6:,.1f}MM more risked NPV** than rank-by-return-and-cut "
               f"at the same budget + rig limit{gap}. Rig used: {program.rig_used:.0f}/{rig_cap}.")

    l, r = st.columns(2)
    with l:
        st.subheader("Projects by capital efficiency (green = funded)")
        d = econ.sort_values("capital_efficiency", ascending=False).copy()
        d["sel"] = d["project_id"].isin(sel)
        fig = go.Figure(go.Bar(
            x=d["project_id"], y=d["capital_efficiency"],
            marker_color=["#2ca02c" if s else "#c9c9c9" for s in d["sel"]]))
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="project (ranked)", yaxis_title="risked NPV / capex (x)",
                          xaxis=dict(showticklabels=False))
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.subheader("Allocation by category")
        bc = pd.DataFrame([{"Category": k, "Capex $MM": v["capex"]/1e6, "Risked NPV $MM": v["risked_npv"]/1e6,
                            "n": v["n"]} for k, v in program.by_category.items()])
        if len(bc):
            pf = go.Figure()
            pf.add_bar(x=bc["Category"], y=bc["Capex $MM"], name="Capex $MM", marker_color="#1F3A5F")
            pf.add_bar(x=bc["Category"], y=bc["Risked NPV $MM"], name="Risked NPV $MM", marker_color="#4F81BD")
            pf.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), barmode="group",
                             legend=dict(orientation="h"))
            st.plotly_chart(pf, use_container_width=True)

    fl, fr = st.columns(2)
    with fl:
        st.subheader("Efficient frontier — NPV vs. budget")
        front = budget_frontier(projects, price, total_capex, rig_cap, steps=14)
        ff = go.Figure(go.Scatter(x=front["budget"]/1e6, y=front["risked_npv"]/1e6, mode="lines+markers",
                                  line=dict(color="#1F3A5F")))
        ff.add_vline(x=budget/1e6, line_dash="dash", line_color="#2ca02c",
                     annotation_text=f"budget ${budget/1e6:,.0f}MM")
        ff.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                         xaxis_title="capital budget ($MM)", yaxis_title="optimal risked NPV ($MM)")
        st.plotly_chart(ff, use_container_width=True)
        st.caption("The curve flattens — the marginal value of capital diminishes. That's the picture that "
                   "sizes (or caps) the budget ask.")
    with fr:
        st.subheader("Price-deck sensitivity")
        ps = price_scenarios(projects, budget, rig_cap)
        pp = go.Figure(go.Bar(x=[f"${p:.0f}" for p in ps["price"]], y=ps["risked_npv"]/1e6,
                              marker_color="#4F81BD"))
        pp.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                         xaxis_title="realized price", yaxis_title="program risked NPV ($MM)")
        st.plotly_chart(pp, use_container_width=True)

    st.subheader("Recommended program")
    pt = econ[econ["project_id"].isin(sel)].sort_values("risked_npv_usd", ascending=False).copy()
    pt["Capex"] = pt["capex_usd"].map(lambda v: f"${v/1e6:,.2f}MM")
    pt["Risked NPV"] = pt["risked_npv_usd"].map(lambda v: f"${v/1e6:,.2f}MM")
    pt["Cap. eff."] = pt["capital_efficiency"].map(lambda v: f"{v:.2f}x")
    pt["IRR"] = pt["irr_pct"].map(lambda v: f"{v:.0f}%" if pd.notna(v) else "—")
    pt["Pc"] = pt["pc"].map(lambda v: f"{v:.0%}")
    st.dataframe(pt[["project_id", "name", "label", "area", "Capex", "Risked NPV", "Cap. eff.", "IRR", "Pc"]]
                 .rename(columns={"project_id": "ID", "name": "Project", "label": "Type", "area": "Area"}),
                 use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📝 Capital-program memo")
    if st.button("Generate memo", type="primary"):
        try:
            client = None
            if byok_key:
                from anthropic import Anthropic
                client = Anthropic(api_key=byok_key)
            with st.spinner("Writing the capital-committee memo…"):
                md = write_memo(program, greedy, ps, program.by_category, client=client)
            st.markdown(md)
        except MissingAPIKey:
            st.info("No API key — showing the deterministic memo. Add your Anthropic key in the sidebar "
                    "for the narrated version.")
            st.markdown(render_memo_markdown(program, greedy, ps, program.by_category))

with tab_sched:
    st.subheader("Quarterly schedule (per-quarter rig capacity)")
    rig_q = st.slider("Rig-day capacity per quarter", 20, rig_cap, max(rig_cap // 4, 20), 5)
    sched = schedule_program(econ, list(sel), projects, n_quarters=4, rig_per_quarter=rig_q)
    if len(sched):
        agg = sched.groupby("quarter").agg(capex=("capex_usd", "sum"), rig=("rig_days", "sum"),
                                           npv=("risked_npv_usd", "sum"), n=("project_id", "size")).reset_index()
        sf = go.Figure()
        sf.add_bar(x=agg["quarter"], y=agg["capex"]/1e6, name="Capex $MM", marker_color="#1F3A5F")
        sf.add_bar(x=agg["quarter"], y=agg["rig"], name="Rig-days", marker_color="#C0504D", yaxis="y2")
        sf.update_layout(height=320, barmode="group", margin=dict(l=0, r=0, t=10, b=0),
                         yaxis2=dict(overlaying="y", side="right", title="rig-days"), legend=dict(orientation="h"))
        st.plotly_chart(sf, use_container_width=True)
        disp = sched.copy()
        disp["capex_usd"] = disp["capex_usd"].map(lambda v: f"${v/1e6:,.2f}MM")
        disp["risked_npv_usd"] = disp["risked_npv_usd"].map(lambda v: f"${v/1e6:,.2f}MM")
        st.dataframe(disp.rename(columns={"name": "Project", "category": "Type", "quarter": "Quarter",
                                          "capex_usd": "Capex", "rig_days": "Rig-days", "risked_npv_usd": "Risked NPV"})
                     [["Quarter", "Project", "Type", "Capex", "Rig-days", "Risked NPV"]],
                     use_container_width=True, hide_index=True)

with tab_val:
    st.subheader("Optimization validation")
    st.caption("The MILP is exact (branch-and-bound). We validate it against the greedy baseline it must "
               "beat-or-tie, and report the LP-relaxation bound — a provable cap on the best possible NPV.")
    v = st.columns(3)
    v[0].metric("MILP risked NPV", f"${program.risked_npv/1e6:,.1f}MM")
    v[1].metric("Greedy baseline", f"${greedy.risked_npv/1e6:,.1f}MM", f"+${uplift/1e6:,.1f}MM optimizer")
    v[2].metric("LP bound (optimality)", f"${(program.lp_bound or 0)/1e6:,.1f}MM",
                f"{program.optimality_gap_pct:.2f}% gap" if program.optimality_gap_pct is not None else "—")
    st.markdown(
        f"- **Budget:** ${program.capex_used/1e6:,.1f}MM used of ${budget/1e6:,.0f}MM "
        f"({program.capex_used/budget*100:.0f}%) — feasible ✓\n"
        f"- **Rig-days:** {program.rig_used:.0f} of {rig_cap} — feasible ✓\n"
        f"- **Optimizer uplift:** ${uplift/1e6:,.1f}MM ({uplift/max(greedy.risked_npv,1)*100:.1f}%) over "
        f"rank-by-return-and-cut — the value of optimizing both scarce resources (capital **and** rigs) "
        f"jointly instead of ranking on one.")
