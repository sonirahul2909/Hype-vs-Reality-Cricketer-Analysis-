"""Stage 4: percentiles and the Overhyped / Fairly Rated / Underrated verdict.

The verdict is a RULE, not a learned truth:
    gap = hype percentile - impact percentile
    gap >= +20 -> Overhyped,  gap <= -20 -> Underrated,  otherwise Fairly Rated

Fairness rules:
  * impact is ranked only against players of the same (format, competition, role group);
  * 'All' formats / competitions: each group is ranked separately, then combined weighted by innings;
  * Bowlers get no verdict (the model scores batting only).
"""
import numpy as np
import pandas as pd

MIN_INNINGS = {"Test": 8, "ODI": 10, "T20": 12}
GAP_CUT = 20
MIN_COHORT = 15
MIN_GROUP = 8                      # smallest group that is ranked at all
VERDICT_ROLES = ("Batter", "All-rounder")


def window_start(inn, years):
    end = inn["date"].max()
    return end - pd.DateOffset(years=years) if years else inn["date"].min()


def _pct(s):
    return s.rank(pct=True) * 100


def cohort_table(inn, hype_monthly, fmt, comp, years, roles):
    """One row per player for this (format, competition, window): impact, hype, percentiles, gap."""
    start = window_start(inn, years)
    w = inn[inn["date"] >= start]
    if fmt != "All":
        w = w[w["fmt"] == fmt]
    if comp != "All":
        w = w[w["comp"] == comp]
    w = w.merge(roles[["batter", "role_group"]], on="batter", how="inner")
    w = w[w["role_group"].isin(VERDICT_ROLES)]
    g = (w.groupby(["batter", "fmt", "comp", "role_group"])
         .agg(innings=("impact", "size"), isum=("impact", "sum")).reset_index())
    g = g[g["innings"] >= g["fmt"].map(MIN_INNINGS)]
    grp = ["fmt", "comp", "role_group"]
    g = g[g.groupby(grp)["batter"].transform("size") >= MIN_GROUP].copy()
    g["impact"] = g["isum"] / g["innings"]
    g["pct"] = g.groupby(grp)["impact"].rank(pct=True) * 100
    g["wp"] = g["pct"] * g["innings"]
    per = (g.groupby("batter").agg(innings=("innings", "sum"), isum=("isum", "sum"),
                                   wp=("wp", "sum"), role_group=("role_group", "first")).reset_index())
    per["impact"] = per["isum"] / per["innings"]
    per["impact_pct"] = _pct(per["wp"] / per["innings"])
    per = per.drop(columns=["isum", "wp"])

    if hype_monthly is None or len(hype_monthly) == 0:  # impact-only mode
        per["hype_raw"], per["hype_pct"], per["gap"], per["verdict"] = np.nan, np.nan, np.nan, None
        return per
    h = hype_monthly[(hype_monthly["month"] >= start.to_period("M").to_timestamp())]
    hype = (h.assign(lv=np.log1p(h["views"])).groupby("player")["lv"].mean()
            .rename("hype_raw").reset_index().rename(columns={"player": "batter"}))
    t = per.merge(hype, on="batter", how="inner")
    if len(t):
        t["hype_pct"] = _pct(t["hype_raw"])
        t["gap"] = t["hype_pct"] - t["impact_pct"]
        t["verdict"] = np.select([t["gap"] >= GAP_CUT, t["gap"] <= -GAP_CUT],
                                 ["Overhyped", "Underrated"], "Fairly Rated")
    else:
        t = t.assign(hype_pct=np.nan, gap=np.nan, verdict=None)
    return t


COMP_TEXT = {"All": "", "International": " (international matches)", "League": " (league matches such as the IPL)"}


def summary_text(row, fmt, comp, years, tone=None):
    win = f"the last {years} year{'s' if years > 1 else ''}" if years else "all available data"
    fmt_s = "all formats" if fmt == "All" else fmt
    base = (f"In {fmt_s}{COMP_TEXT.get(comp, '')} over {win}, public attention puts him at the "
            f"{row['hype_pct']:.0f}th percentile, while his pressure-weighted batting impact "
            f"({row['impact']:.1f} per innings across {int(row['innings'])} innings) is at the "
            f"{row['impact_pct']:.0f}th percentile among {row['role_group'].lower()}s.")
    if row["verdict"] == "Overhyped":
        base += f" Attention runs {row['gap']:.0f} points ahead of on-field impact."
    elif row["verdict"] == "Underrated":
        base += f" On-field impact runs {-row['gap']:.0f} points ahead of the attention he gets."
    else:
        base += " Attention and impact are well matched."
    if tone is not None and tone.get("n", 0) >= 10 and tone.get("tone") is not None:
        mood = "positive" if tone["tone"] > 0.1 else ("negative" if tone["tone"] < -0.1 else "neutral")
        base += f" Recent headline mood is {mood} ({tone['n']} headlines)."
    return base
