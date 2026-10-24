"""Generate a SYNTHETIC data set (fictional players) in the real Cricsheet JSON
format, so the whole pipeline and app can be tested without internet.

    python make_demo_data.py && python build_data.py --demo
Nothing here is real cricket data; the app shows a DEMO banner for it.
"""
import json
import os
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

random.seed(7)
np.random.seed(7)
OUT = "data_demo"
FIRST = ["Aarav", "Bryn", "Caleb", "Dev", "Eli", "Farid", "Gus", "Hiro", "Ivan", "Jon",
         "Kai", "Leo", "Milo", "Nils", "Omar", "Pavel", "Quin", "Rafe", "Sven", "Tomas"]
LAST = ["Ashby", "Brandt", "Corva", "Dunmore", "Ellery", "Fenwick", "Garrow", "Holt",
        "Ibarra", "Jarvis", "Kestrel", "Lund", "Marlow", "Norcott", "Okoye"]
TYPES = ["Pace", "Pace", "Off-spin", "Leg-spin", "Left-arm spin"]


def make_teams(n=14):
    names = [f"{f[0]} {l}" for f in FIRST for l in LAST]
    random.shuffle(names)
    teams = {}
    for t in range(n):
        squad = []
        for i in range(11):
            nm = names.pop()
            bat = max(0.4, np.random.normal(1.0, 0.15)) if i < 7 else 0.55
            squad.append({"name": nm, "bat": bat,
                          "pace": np.clip(np.random.normal(1, .2), .6, 1.4),
                          "spin": np.clip(np.random.normal(1, .2), .6, 1.4),
                          "bowl": np.random.uniform(.75, 1.3) if i >= 4 else 3.0,
                          "type": random.choice(TYPES),
                          "clutch": np.clip(np.random.normal(1, .15), .7, 1.3)})
        teams[f"Team {chr(65 + t)}"] = squad
    return teams


PARAMS = {  # base wicket prob per ball, run weights for 0,1,2,3,4,6
    "T20": (0.072, [.36, .36, .07, .01, .13, .07], 20),
    "ODI": (0.034, [.42, .38, .08, .01, .08, .03], 50),
    "Test": (0.0185, [.66, .21, .04, .005, .08, .01], 90),
}


def sim_innings(bat, bowl, fmt, target=None, max_ov=None):
    wp, rw, mo = PARAMS[fmt]
    max_ov = max_ov or mo
    bowlers = [p for p in bowl if p["bowl"] < 2]
    order = list(bat)
    striker, non = order[0], order[1]
    nxt, wkts, score, overs = 2, 0, 0, []
    for ov in range(max_ov):
        b = random.choice(bowlers)
        dl_list, legal = [], 0
        while legal < 6:
            tgt_pressure = 1.0
            if target is not None:
                need = target - score
                left = max((max_ov - ov) * 6 - legal, 1)
                tgt_pressure = np.clip((need / left) / (target / (max_ov * 6)), .7, 2.0)
            skill = striker["bat"] * (striker["pace"] if b["type"] == "Pace" else striker["spin"])
            if tgt_pressure > 1.15 or wkts >= 4:
                skill *= striker["clutch"]
            ex = random.random()
            wide, noball = ex < .02, .02 <= ex < .03
            runs_bat = 0
            if not wide:
                w = np.array(rw)
                w = w * np.array([1 / skill, 1, 1, 1, skill, skill])
                if fmt != "Test" and tgt_pressure > 1:
                    w[4:] *= tgt_pressure ** .8
                runs_bat = int(random.choices([0, 1, 2, 3, 4, 6], weights=w / w.sum())[0])
            d = {"batter": striker["name"], "bowler": b["name"], "non_striker": non["name"],
                 "runs": {"batter": runs_bat, "extras": 1 if (wide or noball) else 0,
                          "total": runs_bat + (1 if (wide or noball) else 0)}}
            if wide:
                d["extras"] = {"wides": 1}
            if noball:
                d["extras"] = {"noballs": 1}
            wicket = (not wide) and random.random() < wp * b["bowl"] / skill * (
                1.3 if (fmt != "Test" and tgt_pressure > 1.3) else 1)
            if wicket:
                kind = random.choices(["bowled", "caught", "lbw", "run out", "stumped"],
                                      weights=[.2, .55, .12, .08, .05])[0]
                d["wickets"] = [{"player_out": striker["name"], "kind": kind}]
                wkts += 1
            dl_list.append(d)
            score += d["runs"]["total"]
            if not (wide or noball):
                legal += 1
            if wicket:
                if wkts >= 10 or nxt >= 11:
                    overs.append({"over": ov, "deliveries": dl_list})
                    return overs, score, wkts
                striker = order[nxt]
                nxt += 1
            elif runs_bat % 2 == 1:
                striker, non = non, striker
            if target is not None and score >= target:
                overs.append({"over": ov, "deliveries": dl_list})
                return overs, score, wkts
        overs.append({"over": ov, "deliveries": dl_list})
        striker, non = non, striker
    return overs, score, wkts


def make_match(i, teams, fmt, d):
    a, b = random.sample(list(teams), 2)
    sa, sb = teams[a], teams[b]
    info = {"balls_per_over": 6, "dates": [d.isoformat()], "gender": "male", "match_type": fmt,
            "teams": [a, b], "event": {"name": "Demo League"}}
    if fmt == "Test":
        o1, s1, _ = sim_innings(sa, sb, fmt)
        o2, s2, _ = sim_innings(sb, sa, fmt)
        o3, s3, _ = sim_innings(sa, sb, fmt)
        need = s1 + s3 - s2 + 1
        inns = [{"team": a, "overs": o1}, {"team": b, "overs": o2}, {"team": a, "overs": o3}]
        if need > 0:
            o4, s4, _ = sim_innings(sb, sa, fmt, target=need)
            inns.append({"team": b, "overs": o4})
            win = b if s4 >= need else a
            info["outcome"] = {"winner": win}
        else:
            info["outcome"] = {"winner": a}
    else:
        o1, s1, _ = sim_innings(sa, sb, fmt)
        o2, s2, _ = sim_innings(sb, sa, fmt, target=s1 + 1)
        inns = [{"team": a, "overs": o1},
                {"team": b, "overs": o2, "target": {"overs": PARAMS[fmt][2], "runs": s1 + 1}}]
        info["outcome"] = {"winner": b if s2 > s1 else a}
    info["players"] = {a: [p["name"] for p in sa], b: [p["name"] for p in sb]}
    return {"meta": {"data_version": "demo"}, "info": info, "innings": inns}


def main():
    teams = make_teams()
    os.makedirs(f"{OUT}/raw", exist_ok=True)
    start = date(2021, 1, 10)
    n = 0
    plan = [("T20", 520), ("ODI", 160), ("Test", 45)]
    total_days = (date(2026, 9, 20) - start).days
    for fmt, count in plan:
        for _ in range(count):
            d = start + timedelta(days=random.randint(0, total_days))
            m = make_match(n, teams, fmt, d)
            with open(f"{OUT}/raw/demo_{n}.json", "w") as f:
                json.dump(m, f)
            n += 1
    print(f"wrote {n} matches")

    # bowler type lookup for the demo players
    rows = [(p["name"], p["type"]) for sq in teams.values() for p in sq if p["bowl"] < 2]
    pd.DataFrame(rows, columns=["bowler", "type"]).to_csv(f"{OUT}/bowler_types.csv", index=False)

    # synthetic monthly "pageviews": popularity partly (not fully) tied to skill
    months = pd.date_range("2018-01-01", "2026-09-01", freq="MS")
    out = []
    for sq in teams.values():
        for p in sq[:8]:
            pop = np.exp(np.random.normal(9.5 + 0.8 * (p["bat"] - 1), 1.1))
            season = 1 + .25 * np.sin(np.arange(len(months)) / 3.0)
            noise = np.random.lognormal(0, .25, len(months))
            for mth, s, nz in zip(months, season, noise):
                out.append((p["name"], mth, int(pop * s * nz / 12), p["name"]))
    pd.DataFrame(out, columns=["player", "month", "views", "wiki_title"]).to_csv(
        f"{OUT}/hype_monthly.csv", index=False)
    print("wrote synthetic hype data")


if __name__ == "__main__":
    main()
