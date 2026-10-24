"""Hype vs Reality: Cricket Player Analyzer  (Streamlit app).   Run: streamlit run app.py"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cricket.profile import build_profile, findings_text
from cricket.scoring import reality_check, situation_text
from cricket.verdict import GAP_CUT, MIN_COHORT, MIN_INNINGS, cohort_table, summary_text, window_start

st.set_page_config(page_title="Hype vs Reality - Cricket", page_icon="🏏", layout="wide")

DATA_DIR = os.environ.get("DATA_DIR") or (
    "data/processed" if os.path.exists("data/processed/innings.parquet") else "data_demo/processed")
WINDOWS = {"Last 1 year": 1, "Last 2 years": 2, "Last 3 years": 3, "Last 5 years": 5, "All data": None}
COMPS = {"All competitions": "All", "International": "International", "Leagues (IPL etc.)": "League"}
BADGE = {"Overhyped": ("#b91c1c", "#fee2e2"), "Fairly Rated": ("#1d4ed8", "#dbeafe"),
         "Underrated": ("#15803d", "#dcfce7")}
C_HYPE, C_IMPACT = "#d97706", "#2563eb"


@st.cache_resource(show_spinner="Loading data ...")
def load(d):
    if not os.path.exists(f"{d}/innings.parquet"):
        return None
    inn = pd.read_parquet(f"{d}/innings.parquet")
    balls = pd.read_parquet(f"{d}/balls.parquet")
    hp = f"{d}/hype_monthly.parquet"
    hype = pd.read_parquet(hp) if os.path.exists(hp) else pd.DataFrame(columns=["player", "month", "views"])
    tone = json.load(open(f"{d}/hype_tone.json")) if os.path.exists(f"{d}/hype_tone.json") else {}
    meta = json.load(open(f"{d}/meta.json"))
    rp = f"{d}/roles.parquet"
    roles = pd.read_parquet(rp) if os.path.exists(rp) else None
    return inn, balls, hype, tone, meta, roles


@st.cache_data(show_spinner=False)
def get_cohort(d, fmt, comp, years):
    inn, _, hype, _, _, roles = load(d)
    return cohort_table(inn, hype, fmt, comp, years, roles)


data = load(DATA_DIR)
st.title("🏏 Hype vs Reality")
st.caption("Does a player's public hype match what he actually does under pressure?")
if data is None:
    st.error("No processed data found. Run `python download_data.py` then `python build_data.py` "
             "(or `python make_demo_data.py && python build_data.py --demo`).")
    st.stop()
inn, balls, hype, tone, meta, roles = data
if roles is None or "comp" not in inn.columns:
    st.error("This data was built with an older version. Run `python build_data.py --skip-hype` again, then restart the app.")
    st.stop()
if meta.get("demo"):
    st.warning("DEMO MODE: all players and numbers on this page are synthetic (fictional). "
               "Build the real data set to analyse real players.")
has_hype = len(hype) > 0
if not has_hype:
    st.info("No hype data in this build, so only impact scores are shown (no verdict).")

# ---------------------------------------------------------------- controls
players = sorted(inn["batter"].unique())
default_player = inn.groupby("batter").size().idxmax()
with st.form("controls"):
    c1, c2, c3, c4, c5 = st.columns([2.6, 1.2, 1.9, 1.6, 1])
    player = c1.selectbox("Player (type to search)", players, index=players.index(default_player))
    fmt = c2.selectbox("Format", ["All", "Test", "ODI", "T20"])
    comp_label = c3.selectbox("Competition", list(COMPS))
    win_label = c4.selectbox("Time window", list(WINDOWS), index=2)
    c5.write("")
    c5.form_submit_button("Analyze", width="stretch", type="primary")
years = WINDOWS[win_label]
comp = COMPS[comp_label]

tab_main, tab_gems, tab_method = st.tabs(["Analyzer", "Hidden Gems leaderboard", "How this works"])

# ---------------------------------------------------------------- main analysis
cohort = get_cohort(DATA_DIR, fmt, comp, years)
start = window_start(inn, years)
w = inn[(inn["date"] >= start) & ((inn["fmt"] == fmt) if fmt != "All" else True) & ((inn["comp"] == comp) if comp != "All" else True) & (inn["batter"] == player)]
row = cohort[cohort["batter"] == player]

# ---------------------------------------------------------------- hidden gems
with tab_gems:
    st.subheader(f"Hidden Gems: {fmt}, {comp_label.lower()}, {win_label.lower()}")
    if not has_hype or cohort.empty or cohort["verdict"].isna().all():
        st.info("Needs hype data and enough players in this format / competition / window.")
    else:
        cols = {"batter": "Player", "role_group": "Role", "innings": "Innings", "impact": "Impact / innings",
                "impact_pct": "Impact pct", "hype_pct": "Hype pct", "gap": "Gap"}
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Most underrated** (impact well above hype)")
            st.dataframe(cohort.sort_values("gap").head(10)[list(cols)].rename(columns=cols).round(1),
                         hide_index=True, width="stretch")
        with g2:
            st.markdown("**Most overhyped** (hype well above impact)")
            st.dataframe(cohort.sort_values("gap", ascending=False).head(10)[list(cols)].rename(columns=cols).round(1),
                         hide_index=True, width="stretch")

# ---------------------------------------------------------------- method
with tab_method:
    st.markdown(f"""
**Data.** Cricsheet ball-by-ball files (men's matches). Built {meta.get('built')}; latest match {meta.get('latest_match')}.
{meta['players']} tracked players, {meta['matches']:,} matches.

**Impact.** Every ball gets a pressure number (about 1.0 normally). It goes up when his team has lost more wickets than
usual for that stage, and, when chasing, when the required run rate is high. Each run is multiplied by the pressure of
the ball it was scored on, and summed per innings. Impact per innings is ranked only against tracked players of the same role group (batter or all-rounder), in the same format and competition. With "All" selected, each format/competition is ranked separately and then combined, weighted by innings.

**Hype.** Monthly Wikipedia pageviews (attention), ranked against the other tracked players. Headline mood (VADER on recent
GDELT headlines) is shown in the summary when available, but GDELT only covers about the last 3 months.

**Verdict.** Hype percentile minus impact percentile. A gap of +{GAP_CUT} or more is *Overhyped*, -{GAP_CUT} or less is
*Underrated*. This is a transparent rule, not a learned truth: nobody has a ground-truth label for "overhyped".

**Honest limits.**
- Technique and temperament are *proxies* built from scoring patterns, not measurements of footwork or character.
- Pageviews measure attention, and attention can be negative; they are a rough stand-in for hype.
- **Batting only.** Bowlers get no verdict. All-rounders are judged on batting alone, which understates them.
- **Roles are inferred** from where he bats and how much he bowls (last 3 years), not from an official role list.
- **International and league (IPL etc.) cricket are ranked separately.** Hype (Wikipedia pageviews) cannot be split by competition, so a league star's hype includes his IPL fame even when you look at international impact.
- Small samples are noisy; the app warns when a window has few innings.
- Bowler types come from a hand-made lookup, so some balls are "Unknown".
- Check each data source's licence before any commercial use.
""")


with tab_main:
    if w.empty:
        st.warning(f"{player} has no recorded innings in {fmt} ({comp_label.lower()}) for this window. Try a longer window, another format or another competition.")
        st.stop()

    # ---- Section A: verdict
    rinfo = roles[roles["batter"] == player]
    role_group = rinfo.iloc[0]["role_group"] if len(rinfo) else "Batter"
    sub_role = rinfo.iloc[0]["sub_role"] if len(rinfo) else "Batter"
    st.subheader(f"{player}  -  {fmt}, {comp_label.lower()}, {win_label.lower()}")
    st.caption(f"Role (from his last 3 years): {sub_role}. This app scores batting only.")
    if len(row) and has_hype and row.iloc[0]["verdict"]:
        r = row.iloc[0]
        fg, bg = BADGE[r["verdict"]]
        st.markdown(f"<span style='background:{bg};color:{fg};padding:6px 16px;border-radius:20px;"
                    f"font-weight:700;font-size:1.15rem'>{r['verdict']}</span>", unsafe_allow_html=True)
        st.write("")
        st.write(summary_text(r, fmt, comp, years, tone.get(player)))
        m1, m2, m3 = st.columns(3)
        m1.metric("Hype percentile", f"{r['hype_pct']:.0f}")
        m2.metric("Impact percentile", f"{r['impact_pct']:.0f}")
        m3.metric("Gap (hype - impact)", f"{r['gap']:+.0f}")
        mi = MIN_INNINGS.get(fmt, 12)
        if r["innings"] < 1.5 * mi:
            st.caption(f"Small sample: only {int(r['innings'])} innings. Treat this verdict gently.")
        if len(cohort) < MIN_COHORT:
            st.caption(f"Small comparison group ({len(cohort)} players), so percentiles are rough.")
        st.caption(f"Percentiles are within the same format, competition and role group ({role_group}). "
                   f"Rule: gap >= +{GAP_CUT} Overhyped, <= -{GAP_CUT} Underrated, otherwise Fairly Rated.")
    else:
        mi = MIN_INNINGS.get(fmt, 12)
        if role_group == "Bowler":
            why = "he is classed as a bowler and this app scores batting only (a bowling model is future work)"
        elif not has_hype:
            why = "this build has no hype data"
        elif len(w) < mi:
            why = f"needs at least {mi} innings in this window (has {len(w)})"
        else:
            why = "no hype data was found for this player (check wiki_titles_used.csv)"
        st.info(f"No verdict: {why}. Impact numbers are still shown below.")

    # ---- Section B: hype vs reality over time
    st.markdown("#### Hype vs Reality over time")
    if has_hype and len(row):
        hp = hype[hype["player"].isin(cohort["batter"])].copy()
        hp["lv"] = np.log1p(hp["views"])
        hp["pct"] = hp.groupby("month")["lv"].rank(pct=True) * 100
        hs = hp[(hp["player"] == player) & (hp["month"] >= start.to_period("M").to_timestamp())]
        months = pd.date_range(start.to_period("M").to_timestamp(), inn["date"].max(), freq="MS")
        pi = w.set_index("date").sort_index()
        dist = np.sort(cohort["impact"].to_numpy())
        roll = []
        for m in months:
            x = pi[(pi.index > m - pd.DateOffset(months=5, days=1)) & (pi.index < m + pd.DateOffset(months=1))]
            roll.append(100 * np.searchsorted(dist, x["impact"].mean()) / len(dist) if len(x) >= 2 else np.nan)
        roll = pd.Series(roll, index=months)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hs["month"], y=hs["pct"], name="Hype (pageview percentile)",
                                 mode="lines", line=dict(color=C_HYPE, width=3)))
        fig.add_trace(go.Scatter(x=roll.index, y=roll.values, name="Reality (6-month impact percentile)",
                                 mode="lines", line=dict(color=C_IMPACT, width=3), connectgaps=False))
        top = w.sort_values("impact", ascending=False).head(5)
        mx, my, mt = [], [], []
        for _, t in top.iterrows():
            mm = t["date"].to_period("M").to_timestamp()
            if mm in roll.index and pd.notna(roll[mm]):
                mx.append(mm); my.append(roll[mm])
                mt.append(f"{t['runs']:.0f} ({t['balls']:.0f}) vs {t['opp']}, {t['date']:%d %b %Y}")
        fig.add_trace(go.Scatter(x=mx, y=my, mode="markers", name="Big innings", text=mt,
                                 hovertemplate="%{text}<extra></extra>",
                                 marker=dict(symbol="diamond", size=12, color="#111827",
                                             line=dict(color="white", width=1.5))))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Percentile (0-100)",
                          yaxis=dict(range=[0, 100]), legend=dict(orientation="h", y=1.12),
                          hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig, width="stretch")
        st.caption("Orange = attention vs the other tracked players that month. Blue = impact in a rolling "
                   "6-month window, ranked against everyone's whole-window impact. Diamonds mark his biggest innings.")
    else:
        st.caption("Chart needs hype data and enough innings in this window.")

    # ---- Section C: clutch breakdown
    st.markdown("#### Clutch Breakdown: best pressure innings")
    top = w.sort_values("impact", ascending=False).head(10).copy()
    top["Situation"] = top.apply(situation_text, axis=1)
    top["Reality check"] = top.apply(reality_check, axis=1)
    top["Score"] = top.apply(lambda r: f"{r['runs']:.0f}{'' if r['out'] else '*'} ({r['balls']:.0f})", axis=1)
    show = pd.DataFrame({"Date": top["date"].dt.strftime("%d %b %Y"), "Format": top["fmt"],
                         "Against": top["opp"], "Situation when he came in": top["Situation"],
                         "Runs (balls)": top["Score"], "Strike rate": top["sr"].round(0),
                         "Avg pressure": top["avg_pressure"].round(2), "Impact": top["impact"].round(1),
                         "Reality check": top["Reality check"]})
    st.dataframe(show, hide_index=True, width="stretch")
    a1, a2, a3 = st.columns(3)
    a1.metric("Innings in window", len(w))
    a2.metric("Avg impact per innings", f"{w['impact'].mean():.1f}")
    hi = w[w["avg_pressure"] >= 1.2]
    a3.metric("Runs per high-pressure innings", f"{hi['runs'].mean():.1f}" if len(hi) >= 3 else "n/a",
              help="Innings whose average pressure was >= 1.2")

    # ---- Section D: strengths & weaknesses
    st.markdown("#### Strengths and weaknesses")
    from cricket.profile import attach_bowler_type  # noqa: F401  (types were attached at build time)
    pb = balls[(balls["batter"] == player) & (balls["date"] >= start)]
    if fmt != "All":
        pb = pb[pb["fmt"] == fmt]
    if comp != "All":
        pb = pb[pb["comp"] == comp]
    prof = build_profile(pb)
    s, wk = findings_text(prof) if fmt != "All" else ([], [])
    cs, cw = st.columns(2)
    with cs:
        st.markdown("**Strengths**")
        for t in s or ["Nothing stands out yet with this much data."]:
            st.write("- " + t)
    with cw:
        st.markdown("**Weaknesses**")
        for t in wk or ["Nothing stands out yet with this much data."]:
            st.write("- " + t)
    if fmt == "All":
        st.caption("Pick a single format to get written strengths and weaknesses (phases differ between formats).")
    if prof["known_share"] < 0.6:
        st.caption(f"Bowler type is known for only {100 * prof['known_share']:.0f}% of balls faced, "
                   "so bowler-type findings are partial.")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Temperament (proxy)**")
        tr = prof.get("temperament_ratio")
        if tr is not None and pd.notna(tr):
            st.metric("Strike rate: high pressure vs low pressure", f"{prof['sr_high_pressure']:.0f} vs {prof['sr_low_pressure']:.0f}",
                      delta=f"{100 * (tr - 1):+.0f}% under pressure")
        else:
            st.write("Not enough balls in both situations.")
        st.caption("A proxy: how his scoring changes when pressure is high. Not a psychological measurement.")
    with p2:
        st.markdown("**Technique balance (proxy)**")
        tb = prof.get("technique_balance")
        if tb is not None and pd.notna(tb):
            st.metric("Pace vs spin balance (100 = equal)", f"{tb:.0f}")
        else:
            st.write("Not enough balls against both pace and spin.")
        st.caption("A proxy: how evenly he scores against pace and spin. Real technique needs tracking data.")

    t1, t2, t3 = st.columns(3)
    with t1:
        if fmt != "All" and len(prof["phase"]):
            st.markdown("**By phase**")
            st.dataframe(prof["phase"][["phase", "balls", "sr", "avg", "outs"]].round(1),
                         hide_index=True, width="stretch")
    with t2:
        if len(prof["bowler_type"]):
            st.markdown("**By bowler type**")
            st.dataframe(prof["bowler_type"][["bowler_type", "balls", "sr", "avg", "outs"]].round(1),
                         hide_index=True, width="stretch")
    with t3:
        if len(prof["dismissal_kinds"]):
            st.markdown("**How he gets out**")
            st.dataframe(prof["dismissal_kinds"], hide_index=True, width="stretch")
        if len(prof["nemesis"]):
            st.markdown("**Bowlers who got him 2+ times**")
            st.dataframe(prof["nemesis"], hide_index=True, width="stretch")
