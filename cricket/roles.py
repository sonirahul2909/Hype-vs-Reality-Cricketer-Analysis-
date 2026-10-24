"""Role classification, so batters, all-rounders and bowlers are not scored on one scale.

Rules (computed from the last 3 years, all formats and competitions together):
  * avg_entry_wkts : average wickets down when he walks in to bat (0 = opener, 8+ = tail-ender)
  * bowl_ratio     : balls he bowls divided by balls he faces
  Bowler       = bats low down (avg entry >= 6 wickets) AND bowls a lot (ratio >= 0.4)
  All-rounder  = bats higher up AND bowls a lot
  Batter       = everyone else (part-time bowlers and wicket-keepers stay batters)
This app only measures BATTING impact, so Bowlers get no verdict.
"""
import numpy as np
import pandas as pd

TAIL_ENTRY = 6.0
BOWL_RATIO = 0.4
OPENER_ENTRY = 1.0
MIDDLE_ENTRY = 3.5


def classify_roles(balls, inn, tracked, latest, years=3):
    start = latest - pd.DateOffset(years=years)
    rb = balls[balls["date"] >= start]
    bowled = rb[rb["legal"]].groupby("bowler").size()
    faced = rb.groupby("batter")["faced"].sum()
    ri = inn[inn["date"] >= start]
    r = pd.DataFrame(index=pd.Index(list(tracked), name="batter"))
    r["avg_entry_wkts"] = ri.groupby("batter")["entry_wkts"].mean().reindex(r.index)
    r["balls_bowled"] = bowled.reindex(r.index).fillna(0)
    r["balls_faced"] = faced.reindex(r.index).fillna(0)
    r["bowl_ratio"] = r["balls_bowled"] / r["balls_faced"].clip(lower=1)
    bowls = r["bowl_ratio"] >= BOWL_RATIO
    tail = r["avg_entry_wkts"] >= TAIL_ENTRY
    r["role_group"] = np.select([tail & bowls, (~tail) & bowls], ["Bowler", "All-rounder"], "Batter")
    r["sub_role"] = np.select([r["avg_entry_wkts"] < OPENER_ENTRY, r["avg_entry_wkts"] < MIDDLE_ENTRY],
                              ["Opener", "Top / middle order"], "Lower order")
    r.loc[r["role_group"] == "Bowler", "sub_role"] = "Bowler"
    r.loc[r["role_group"] == "All-rounder", "sub_role"] = "All-rounder"
    return r.reset_index()
