"""Stage 2: strengths, weaknesses and the two proxy metrics."""
import numpy as np
import pandas as pd

from .parser import BOWLER_KINDS
from .scoring import HIGH_P, LOW_P

MIN_BALLS = 60  # ignore groups smaller than this when writing sentences


def load_bowler_types(path):
    try:
        t = pd.read_csv(path)
        return dict(zip(t["bowler"], t["type"]))
    except FileNotFoundError:
        return {}


def attach_bowler_type(b, lookup):
    b = b.copy()
    b["bowler_type"] = b["bowler"].map(lookup).fillna("Unknown")
    b["bowler_kind"] = b["bowler_type"].map(
        lambda t: "Pace" if t == "Pace" else ("Unknown" if t == "Unknown" else "Spin"))
    return b


def _group(df, col):
    g = df.groupby(col).agg(runs=("runs", "sum"), balls=("faced", "sum"),
                            outs=("dismissed", "sum")).reset_index()
    g["sr"] = np.where(g["balls"] > 0, 100 * g["runs"] / g["balls"], np.nan)
    g["avg"] = np.where(g["outs"] > 0, g["runs"] / g["outs"], np.nan)
    g["balls_per_out"] = np.where(g["outs"] > 0, g["balls"] / g["outs"], np.nan)
    return g


def build_profile(balls: pd.DataFrame) -> dict:
    """balls: one player's balls (already filtered by format/window), with
    bowler_type, bowler_kind and phase columns."""
    out = {"n_balls": int(balls["faced"].sum())}
    tot_runs, tot_outs = balls["runs"].sum(), balls["dismissed"].sum()
    out["sr"] = 100 * tot_runs / max(out["n_balls"], 1)
    out["avg"] = tot_runs / tot_outs if tot_outs else np.nan
    out["bpo"] = out["n_balls"] / tot_outs if tot_outs else np.nan

    out["phase"] = _group(balls, "phase")
    known = balls[balls["bowler_type"] != "Unknown"]
    out["bowler_type"] = _group(known, "bowler_type") if len(known) else pd.DataFrame()
    out["bowler_kind"] = _group(known, "bowler_kind") if len(known) else pd.DataFrame()
    out["known_share"] = len(known) / max(len(balls), 1)

    d = balls[balls["dismissed"]]
    out["dismissal_kinds"] = (d["kind"].value_counts().rename_axis("kind")
                              .reset_index(name="count"))
    dd = d[d["kind"].isin(BOWLER_KINDS)]
    nem = (dd.groupby("bowler").size().sort_values(ascending=False).head(5)
           .rename("dismissals").reset_index())
    out["nemesis"] = nem[nem["dismissals"] >= 2]

    # ---- temperament proxy: strike rate under high vs low pressure ----
    hi, lo = balls[balls["pressure"] >= HIGH_P], balls[balls["pressure"] < LOW_P]
    if len(hi) >= 40 and len(lo) >= 40:
        sr_hi, sr_lo = 100 * hi["runs"].sum() / max(hi["faced"].sum(), 1), \
                       100 * lo["runs"].sum() / max(lo["faced"].sum(), 1)
        out["temperament_ratio"] = sr_hi / sr_lo if sr_lo > 0 else np.nan
        out["sr_high_pressure"], out["sr_low_pressure"] = sr_hi, sr_lo
    else:
        out["temperament_ratio"] = np.nan

    # ---- technique proxy: balance between pace and spin ----
    bk = out["bowler_kind"]
    if len(bk) and {"Pace", "Spin"} <= set(bk["bowler_kind"]):
        r = bk.set_index("bowler_kind")
        if r.loc["Pace", "balls"] >= 40 and r.loc["Spin", "balls"] >= 40:
            a, s = r.loc["Pace", "sr"], r.loc["Spin", "sr"]
            out["pace_spin_gap"] = abs(a - s) / max(a, s)
            out["technique_balance"] = 100 * (1 - min(out["pace_spin_gap"], 1))
    out.setdefault("technique_balance", np.nan)
    return out


def findings_text(p: dict):
    """Turn the tables into plain-English strengths / weaknesses lists."""
    strengths, weaknesses = [], []
    base_sr, base_bpo = p["sr"], p["bpo"]
    for key, label in [("phase", "in the {} phase"), ("bowler_type", "against {}")]:
        t = p[key]
        if not len(t):
            continue
        for _, r in t[t["balls"] >= MIN_BALLS].iterrows():
            if r["bowler_type"if key == "bowler_type" else "phase"] == "Unknown":
                continue
            name = r["bowler_type" if key == "bowler_type" else "phase"]
            lbl = label.format(name.lower() if key == "bowler_type" else name.lower())
            slow = r["sr"] < 0.85 * base_sr
            fragile = pd.notna(r["balls_per_out"]) and pd.notna(base_bpo) and r["balls_per_out"] < 0.6 * base_bpo
            quick = r["sr"] > 1.15 * base_sr
            solid = pd.isna(r["balls_per_out"]) or (pd.notna(base_bpo) and r["balls_per_out"] > 1.5 * base_bpo)
            if slow or fragile:
                weaknesses.append(f"Weaker {lbl} (SR {r['sr']:.0f}, "
                                  f"{r['outs']:.0f} dismissals in {r['balls']:.0f} balls)")
            elif quick and solid:
                strengths.append(f"Strong {lbl} (SR {r['sr']:.0f}, "
                                 f"{r['outs']:.0f} dismissals in {r['balls']:.0f} balls)")
    return strengths, weaknesses
