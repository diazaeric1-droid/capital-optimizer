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

import theme

from src import __version__
from src.economics import economics_frame
from src.narrator import MissingAPIKey, render_memo_markdown, write_memo
from src.optimizer import (
    InfeasibleProgram,
    greedy_select,
    greedy_select_multiperiod,
    milp_select_multiperiod,
    optimize,
)
from src.projects import load_projects
from src.scenarios import budget_frontier, price_scenarios
from src.schedule import schedule_program

theme.setup_page("Capital Program Optimizer", icon="🛢️")
theme.suite_nav("capital")
theme.header(
    "Capital Program Optimizer",
    subtitle="Rank a drilling / DUC / workover backlog by risked economics and pick the program that "
             "maximizes NPV under a capital budget + rig capacity. Built by an ex-OXY / ex-Shell Staff PE.",
    chips=[(f"v{__version__}", "ver"), ("MILP optimal", "eval")],
)
theme.data_badge("synthetic", "Modeled drilling / DUC / workover backlog — future capital projects aren't public data.")

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
        "- Deterministic optimization; the LLM only writes the capital-committee memo. Bring your own key.\n"
        "\n"
        "**New in v0.2.0:**\n"
        "- **Multi-period capital plan** — pick *which project runs in which period* under per-period "
        "budget + rig-day capacity (fund-once, earliest-start, period-discounted NPV), with a "
        "project×period **schedule heatmap** + per-period utilization (~12% / $35MM over greedy).\n"
        "- **Infeasibility banner** — infeasible programs are flagged honestly instead of faking a plan.\n"
        "- **Funded-vs-rejected scatter** (capex × risked NPV, sized by rig-days) + a **shared fleet "
        "registry** for consistent Permian field/formation identity across the suite.\n"
        "- **Unified dark/navy theme** + cross-app **suite navigator** in the sidebar."
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
    mode = st.radio("Planning mode",
                    ["📈 Single-period program", "🗓️ Multi-period plan"],
                    help="Single-period picks one program under one budget + rig limit. "
                         "Multi-period schedules the backlog across several periods (e.g. quarters), "
                         "each with its own budget + rig capacity.")
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
if mode.startswith("📈"):
    try:
        program = optimize(econ, budget, rig_cap, float(min_fy))
    except ValueError as exc:
        st.error(str(exc))
        theme.flag("No feasible program", "high")
        st.stop()
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
                marker_color=[theme.GREEN if s else theme.GREY for s in d["sel"]]))
            fig.update_layout(xaxis_title="project (ranked)", yaxis_title="risked NPV / capex (x)",
                              xaxis=dict(showticklabels=False))
            st.plotly_chart(theme.style_fig(fig, height=360), width="stretch")
        with r:
            st.subheader("Allocation by category")
            bc = pd.DataFrame([{"Category": k, "Capex $MM": v["capex"]/1e6, "Risked NPV $MM": v["risked_npv"]/1e6,
                                "n": v["n"]} for k, v in program.by_category.items()])
            if len(bc):
                pf = go.Figure()
                pf.add_bar(x=bc["Category"], y=bc["Capex $MM"], name="Capex $MM", marker_color=theme.NAVY)
                pf.add_bar(x=bc["Category"], y=bc["Risked NPV $MM"], name="Risked NPV $MM", marker_color=theme.BLUE)
                pf.update_layout(barmode="group")
                st.plotly_chart(theme.style_fig(pf, height=360), width="stretch")

        st.subheader("Funded vs. rejected — capex vs. risked NPV (size ∝ rig-days)")
        sc = econ.copy()
        sc["funded"] = sc["project_id"].isin(sel)
        sz = (sc["rig_days"] / sc["rig_days"].max() * 34 + 6) if sc["rig_days"].max() > 0 else 12
        scat = go.Figure()
        for funded, name, color in [(True, "Funded", theme.GREEN), (False, "Rejected", theme.GREY)]:
            g = sc[sc["funded"] == funded]
            if not len(g):
                continue
            scat.add_trace(go.Scatter(
                x=g["capex_usd"] / 1e6, y=g["risked_npv_usd"] / 1e6, mode="markers", name=name,
                marker=dict(color=color, size=sz[g.index], line=dict(width=0.5, color=theme.BG)),
                text=g["project_id"],
                hovertemplate="%{text}<br>capex $%{x:.1f}MM<br>risked NPV $%{y:.1f}MM<extra>" + name + "</extra>"))
        scat.update_layout(xaxis_title="capex ($MM)", yaxis_title="risked NPV ($MM)")
        st.plotly_chart(theme.style_fig(scat, height=340), width="stretch")

        fl, fr = st.columns(2)
        with fl:
            st.subheader("Efficient frontier — NPV vs. budget")
            front = budget_frontier(projects, price, total_capex, rig_cap, steps=14)
            ff = go.Figure(go.Scatter(x=front["budget"]/1e6, y=front["risked_npv"]/1e6, mode="lines+markers",
                                      line=dict(color=theme.NAVY)))
            ff.add_vline(x=budget/1e6, line_dash="dash", line_color=theme.GREEN,
                         annotation_text=f"budget ${budget/1e6:,.0f}MM")
            ff.update_layout(xaxis_title="capital budget ($MM)", yaxis_title="optimal risked NPV ($MM)")
            st.plotly_chart(theme.style_fig(ff, height=320), width="stretch")
            st.caption("The curve flattens — the marginal value of capital diminishes. That's the picture that "
                       "sizes (or caps) the budget ask.")
        with fr:
            st.subheader("Price-deck sensitivity")
            ps = price_scenarios(projects, budget, rig_cap)
            pp = go.Figure(go.Bar(x=[f"${p:.0f}" for p in ps["price"]], y=ps["risked_npv"]/1e6,
                                  marker_color=theme.BLUE))
            pp.update_layout(xaxis_title="realized price", yaxis_title="program risked NPV ($MM)")
            st.plotly_chart(theme.style_fig(pp, height=320), width="stretch")

        st.subheader("Recommended program")
        pt = econ[econ["project_id"].isin(sel)].sort_values("risked_npv_usd", ascending=False).copy()
        pt["Capex"] = pt["capex_usd"].map(lambda v: f"${v/1e6:,.2f}MM")
        pt["Risked NPV"] = pt["risked_npv_usd"].map(lambda v: f"${v/1e6:,.2f}MM")
        pt["Cap. eff."] = pt["capital_efficiency"].map(lambda v: f"{v:.2f}x")
        pt["IRR"] = pt["irr_pct"].map(lambda v: f"{v:.0f}%" if pd.notna(v) else "—")
        pt["Pc"] = pt["pc"].map(lambda v: f"{v:.0%}")
        st.dataframe(pt[["project_id", "name", "label", "area", "Capex", "Risked NPV", "Cap. eff.", "IRR", "Pc"]]
                     .rename(columns={"project_id": "ID", "name": "Project", "label": "Type", "area": "Area"}),
                     width="stretch", hide_index=True)

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
            sf.add_bar(x=agg["quarter"], y=agg["capex"]/1e6, name="Capex $MM", marker_color=theme.NAVY)
            sf.add_bar(x=agg["quarter"], y=agg["rig"], name="Rig-days", marker_color=theme.RED, yaxis="y2")
            sf.update_layout(barmode="group",
                             yaxis2=dict(overlaying="y", side="right", title="rig-days"))
            st.plotly_chart(theme.style_fig(sf, height=320), width="stretch")
            disp = sched.copy()
            disp["capex_usd"] = disp["capex_usd"].map(lambda v: f"${v/1e6:,.2f}MM")
            disp["risked_npv_usd"] = disp["risked_npv_usd"].map(lambda v: f"${v/1e6:,.2f}MM")
            st.dataframe(disp.rename(columns={"name": "Project", "category": "Type", "quarter": "Quarter",
                                              "capex_usd": "Capex", "rig_days": "Rig-days", "risked_npv_usd": "Risked NPV"})
                         [["Quarter", "Project", "Type", "Capex", "Rig-days", "Risked NPV"]],
                         width="stretch", hide_index=True)

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

else:
    # ---------------- Multi-period plan ----------------
    st.subheader("🗓️ Multi-period capital plan")
    st.caption("Assign each backlog project to AT MOST one period (e.g. a quarter). Each period has its "
               "own capital budget and rig-day capacity. The MILP maximizes total (optionally discounted) "
               "risked NPV across the whole horizon — the VP annual-plan deliverable.")

    mc = st.columns(3)
    n_periods = mc[0].number_input("Number of periods", 2, 8, 4, 1,
                                   help="e.g. 4 quarters in an annual plan.")
    n_periods = int(n_periods)
    disc_pct = mc[1].number_input("Per-period discount (%)", 0.0, 25.0, 0.0, 0.5,
                                  help="Discounts risked NPV funded in later periods by 1/(1+r)^t. "
                                       "0% = pure selection/timing.") / 100.0
    even_capex = total_capex / n_periods
    even_rig = total_rig / n_periods

    st.markdown("**Per-period capacity** (defaults split the inventory totals evenly — edit any cell):")
    default_plan = pd.DataFrame({
        "Period": [f"P{t+1}" for t in range(n_periods)],
        "Budget ($MM)": [round(even_capex / 1e6, 1)] * n_periods,
        "Rig-days": [round(even_rig, 0)] * n_periods,
    })
    edited = st.data_editor(default_plan, hide_index=True, width="stretch",
                            disabled=["Period"], key="mp_plan")
    budget_pp = [float(v) * 1e6 for v in edited["Budget ($MM)"]]
    rig_pp = [float(v) for v in edited["Rig-days"]]

    try:
        mp = milp_select_multiperiod(econ, n_periods, budget_pp, rig_pp,
                                     discount_per_period=disc_pct)
    except InfeasibleProgram as exc:
        st.error(str(exc))
        theme.flag("No feasible multi-period plan", "high")
        st.stop()
    mpg = greedy_select_multiperiod(econ, n_periods, budget_pp, rig_pp,
                                    discount_per_period=disc_pct)
    mp_uplift = mp.risked_npv - mpg.risked_npv

    k = st.columns(4)
    k[0].metric("Plan risked NPV", f"${mp.risked_npv/1e6:,.0f}MM",
                help="Discounted total" if disc_pct > 0 else None)
    k[1].metric("Projects funded", f"{mp.n_selected} / {mp.n_available}")
    k[2].metric("vs. greedy", f"+${mp_uplift/1e6:,.1f}MM")
    k[3].metric("Optimality gap",
                f"{mp.optimality_gap_pct:.2f}%" if mp.optimality_gap_pct is not None else "—")

    # project x period schedule heatmap (funded cell = risked NPV $MM)
    st.subheader("Project × period schedule")
    funded_ids = [i for i, _ in mp.selected]
    if funded_ids:
        order = (econ[econ["project_id"].isin(funded_ids)]
                 .sort_values("capital_efficiency", ascending=False)["project_id"].tolist())
        risked_map = dict(zip(econ["project_id"], econ["risked_npv_usd"]))
        place = {i: t for i, t in mp.selected}
        z = []
        for i in order:
            row = [risked_map[i] / 1e6 if place[i] == t else None for t in range(n_periods)]
            z.append(row)
        heat = go.Figure(go.Heatmap(
            z=z, x=[f"P{t+1}" for t in range(n_periods)], y=order,
            colorscale=[[0, theme.PANEL], [1, theme.GREEN]], showscale=True,
            colorbar=dict(title="risked<br>NPV $MM"),
            hovertemplate="%{y} funded in %{x}<br>risked NPV $%{z:.1f}MM<extra></extra>"))
        heat.update_layout(xaxis_title="period", yaxis_title="project (ranked by cap. eff.)")
        st.plotly_chart(theme.style_fig(heat, height=max(320, 22 * len(order))), width="stretch")
    else:
        st.info("No projects funded under the current per-period capacity.")

    # per-period utilization bars
    st.subheader("Per-period utilization")
    ul, ur = st.columns(2)
    pl = [f"P{t+1}" for t in range(n_periods)]
    with ul:
        cf = go.Figure()
        cf.add_bar(x=pl, y=[b / 1e6 for b in budget_pp], name="Budget $MM", marker_color=theme.GREY)
        cf.add_bar(x=pl, y=[c / 1e6 for c in mp.capex_per_period], name="Capex used $MM",
                   marker_color=theme.NAVY)
        cf.update_layout(barmode="overlay", yaxis_title="$MM")
        cf.update_traces(opacity=0.85)
        st.plotly_chart(theme.style_fig(cf, height=300), width="stretch")
        st.caption("Capital deployed vs. budget, each period.")
    with ur:
        rf = go.Figure()
        rf.add_bar(x=pl, y=rig_pp, name="Rig capacity", marker_color=theme.GREY)
        rf.add_bar(x=pl, y=mp.rig_used_per_period, name="Rig-days used", marker_color=theme.RED)
        rf.update_layout(barmode="overlay", yaxis_title="rig-days")
        rf.update_traces(opacity=0.85)
        st.plotly_chart(theme.style_fig(rf, height=300), width="stretch")
        st.caption("Rig-days used vs. capacity, each period.")

    # funded vs rejected scatter (reused from single-period)
    st.subheader("Funded vs. rejected — capex vs. risked NPV (size ∝ rig-days)")
    sc = econ.copy()
    sc["funded"] = sc["project_id"].isin(funded_ids)
    sz = (sc["rig_days"] / sc["rig_days"].max() * 34 + 6) if sc["rig_days"].max() > 0 else 12
    scat = go.Figure()
    for funded, name, color in [(True, "Funded", theme.GREEN), (False, "Rejected", theme.GREY)]:
        g = sc[sc["funded"] == funded]
        if not len(g):
            continue
        scat.add_trace(go.Scatter(
            x=g["capex_usd"] / 1e6, y=g["risked_npv_usd"] / 1e6, mode="markers", name=name,
            marker=dict(color=color, size=sz[g.index], line=dict(width=0.5, color=theme.BG)),
            text=g["project_id"],
            hovertemplate="%{text}<br>capex $%{x:.1f}MM<br>risked NPV $%{y:.1f}MM<extra>" + name + "</extra>"))
    scat.update_layout(xaxis_title="capex ($MM)", yaxis_title="risked NPV ($MM)")
    st.plotly_chart(theme.style_fig(scat, height=340), width="stretch")

    # funded plan table
    st.subheader("Funded plan by period")
    if funded_ids:
        place = {i: t for i, t in mp.selected}
        pt = econ[econ["project_id"].isin(funded_ids)].copy()
        pt["Period"] = pt["project_id"].map(lambda i: f"P{place[i]+1}")
        pt["Capex"] = pt["capex_usd"].map(lambda v: f"${v/1e6:,.2f}MM")
        pt["Risked NPV"] = pt["risked_npv_usd"].map(lambda v: f"${v/1e6:,.2f}MM")
        pt["Cap. eff."] = pt["capital_efficiency"].map(lambda v: f"{v:.2f}x")
        pt["Rig-days"] = pt["rig_days"].map(lambda v: f"{v:.0f}")
        st.dataframe(
            pt.sort_values(["Period", "risked_npv_usd"], ascending=[True, False])
              [["Period", "project_id", "name", "label", "Capex", "Rig-days", "Risked NPV", "Cap. eff."]]
              .rename(columns={"project_id": "ID", "name": "Project", "label": "Type"}),
            width="stretch", hide_index=True)
