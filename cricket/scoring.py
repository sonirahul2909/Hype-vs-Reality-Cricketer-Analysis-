"""Stage 1: pressure per ball, pressure-weighted runs, innings impact.

Pressure P is a number around 1.0 (0.5 = very relaxed, 3.0 = extreme):
    P = wicket_factor x rate_factor
 - wicket_factor: more wickets lost than a normal innings would have lost by
   this stage -> more pressure.
 - rate_factor: when chasing, required run rate vs the "par" rate of the chase.
Every run is multiplied by P of the ball it was scored on.
"""
import numpy as np
import pandas as pd

HIGH_P, LOW_P = 1.15, 1.0  # thresholds for the temperament proxy


def add_pressure(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    lo = b["fmt"] != "Test"
    P = np.ones(len(b))

    # ---- limited overs ----
    if lo.any():
        x = b[lo]
        total_balls = (x["max_overs"] * 6).to_numpy(float)
        f = np.clip(x["legal_before"].to_numpy(float) / total_balls, 0, 1)
        excess = x["wkts_before"].to_numpy(float) - 6.0 * f
        wk_factor = 1 + 0.15 * np.clip(excess, -3, 4)
        rate_factor = np.ones(len(x))
        chasing = x["target"].notna().to_numpy()
        if chasing.any():
            tgt = x["target"].to_numpy(float)
            balls_left = np.maximum(total_balls - x["legal_before"].to_numpy(float), 1)
            req = (tgt - x["score_before"].to_numpy(float)) / (balls_left / 6)
            par = tgt / (total_balls / 6)
            rf = np.clip(req / par, 0.7, 2.0)
            rate_factor = np.where(chasing, rf, 1.0)
        P[lo.to_numpy()] = wk_factor * rate_factor

    # ---- Tests ----
    te = ~lo
    if te.any():
        x = b[te]
        wk_factor = 1 + 0.10 * np.clip(x["wkts_before"].to_numpy(float) - 2, -2, 7)
        chase = np.where(x["target"].notna().to_numpy(), 1.25, 1.0)
        P[te.to_numpy()] = wk_factor * chase

    b["pressure"] = np.clip(P, 0.5, 3.0)
    b["w_runs"] = b["runs"] * b["pressure"]
    return b


def phase_of(fmt, over, legal_before=None):
    """Phase label from the over number (0-based, as in Cricsheet)."""
    over = np.asarray(over)
    if fmt == "T20":
        return np.where(over < 6, "Powerplay", np.where(over < 16, "Middle", "Death"))
    if fmt == "ODI":
        return np.where(over < 10, "Powerplay", np.where(over < 40, "Middle", "Death"))
    return np.where(over < 20, "New ball", np.where(over < 80, "Old ball", "2nd new ball"))


def add_phase(b):
    b = b.copy()
    b["phase"] = ""
    for fmt in b["fmt"].unique():
        m = b["fmt"] == fmt
        b.loc[m, "phase"] = phase_of(fmt, b.loc[m, "over"].to_numpy())
    return b


def innings_table(b: pd.DataFrame, outs: pd.DataFrame) -> pd.DataFrame:
    """One row per batter innings, with the situation when he walked in."""
    keys = ["match_id", "innings_no", "batter"]
    b = b.sort_values(keys + ["legal_before"], kind="stable")
    g = b.groupby(keys, sort=False)
    inn = g.agg(date=("date", "first"), fmt=("fmt", "first"), team=("team", "first"),
                opp=("opp", "first"), runs=("runs", "sum"),
                balls=("faced", "sum"), impact=("w_runs", "sum"),
                avg_pressure=("pressure", "mean"),
                entry_score=("score_before", "first"), entry_wkts=("wkts_before", "first"),
                entry_balls=("legal_before", "first"), target=("target", "first"),
                max_overs=("max_overs", "first"), winner=("winner", "first"),
                team_total=("team_total", "first"),
                comp=("comp", "first")).reset_index()
    inn = inn.merge(outs.drop_duplicates(["match_id", "innings_no", "batter"]),
                    on=keys, how="left")
    inn["out"] = inn["out_kind"].notna()
    inn["team_won"] = inn["winner"] == inn["team"]
    inn["share_of_team"] = np.where(inn["team_total"] > 0, inn["runs"] / inn["team_total"], 0)
    inn["sr"] = np.where(inn["balls"] > 0, 100 * inn["runs"] / inn["balls"], 0)
    return inn


def situation_text(r) -> str:
    """Plain-English 'what was happening when he came in'."""
    if r["fmt"] == "Test":
        s = f"In at {int(r['entry_score'])}/{int(r['entry_wkts'])}"
        if pd.notna(r["target"]):
            s += f", chasing {int(r['target'])}"
        return s
    overs = r["entry_balls"] // 6 + (r["entry_balls"] % 6) / 10
    s = f"In at {int(r['entry_score'])}/{int(r['entry_wkts'])} ({overs:.1f} ov)"
    if pd.notna(r["target"]):
        s += f", chasing {int(r['target'])}"
    return s


def reality_check(r) -> str:
    result = "won" if r["team_won"] else ("lost" if pd.notna(r["winner"]) else "no result")
    share = f"{100 * r['share_of_team']:.0f}% of team runs"
    if r["team_won"] and r["share_of_team"] >= 0.3:
        return f"Match-winning knock - team {result}, {share}"
    if (not r["team_won"]) and pd.notna(r["winner"]) and r["share_of_team"] >= 0.3:
        return f"Lone fight in a lost cause - {share}"
    return f"Team {result}, {share}"
