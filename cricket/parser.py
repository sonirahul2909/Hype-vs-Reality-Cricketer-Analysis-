"""Stage 0: turn Cricsheet ball-by-ball JSON files into one flat table.

One row = one delivery. Each row also carries the match situation *before*
the ball (team score, wickets down, balls bowled, target), which Stage 1 needs.
"""
import json
import os
import re
import zipfile
from datetime import date

import pandas as pd

FORMAT_MAP = {"Test": "Test", "MDM": "Test", "ODI": "ODI", "ODM": "ODI",
              "T20": "T20", "IT20": "T20"}
MAX_OVERS = {"T20": 20, "ODI": 50, "Test": None}
# Dismissals that count as "the bowler got him" (run outs etc. do not)
BOWLER_KINDS = {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}
NOT_OUT_KINDS = {"retired hurt", "retired not out"}


LEAGUE_RE = re.compile(r"league|big bash|blast|smash|hundred|sa20|ilt20|t10|mushtaq|ranji|hazare|county|shield", re.I)


def _competition(info, fmt):
    """International or League. Uses Cricsheet's team_type when present, else the event name."""
    tt = info.get("team_type")
    if tt == "international":
        return "International"
    if tt == "club":
        return "League"
    ev = info.get("event")
    ev = ev.get("name", "") if isinstance(ev, dict) else (ev or "")
    return "League" if LEAGUE_RE.search(ev) else "International"


def _parse_match(data, match_id, men_only=True):
    info = data["info"]
    if men_only and info.get("gender") != "male":
        return [], []
    fmt = FORMAT_MAP.get(info.get("match_type"))
    if fmt is None:
        return [], []
    d = info["dates"][0]
    comp = _competition(info, fmt)
    winner = info.get("outcome", {}).get("winner")
    innings_list = [i for i in data["innings"] if not i.get("super_over")]
    teams = info["teams"]

    # final totals, used for Test chases and team-run shares
    totals = []
    for inn in innings_list:
        t = 0
        for ov in inn.get("overs", []):
            for dl in ov["deliveries"]:
                t += dl["runs"]["total"]
        totals.append(t)

    rows, outs = [], []
    for k, inn in enumerate(innings_list, start=1):
        team = inn["team"]
        opp = teams[1] if teams[0] == team else teams[0]
        target, max_overs = None, MAX_OVERS[fmt]
        if "target" in inn:
            target = inn["target"].get("runs")
            if inn["target"].get("overs"):
                max_overs = float(inn["target"]["overs"])
        elif fmt == "Test" and k == len(innings_list) and k >= 3:
            own_prev = sum(t for j, t in enumerate(totals[:-1])
                           if innings_list[j]["team"] == team)
            opp_tot = sum(t for j, t in enumerate(totals[:-1])
                          if innings_list[j]["team"] != team)
            if opp_tot - own_prev > 0:
                target = opp_tot - own_prev + 1
        score, wkts, legal = 0, 0, 0
        for ov in inn.get("overs", []):
            for dl in ov["deliveries"]:
                ex = dl.get("extras", {})
                wide = "wides" in ex
                is_legal = not (wide or "noballs" in ex)
                dism, kind = False, None
                for w in dl.get("wickets", []):
                    wk = w["kind"]
                    if wk in NOT_OUT_KINDS:
                        continue
                    outs.append((match_id, k, w["player_out"], wk))
                    if w["player_out"] == dl["batter"]:
                        dism, kind = True, wk
                rows.append((match_id, d, fmt, k, team, opp, dl["batter"], dl["bowler"],
                             dl["runs"]["batter"], not wide, is_legal, dism, kind,
                             score, wkts, legal, target, max_overs, winner,
                             ov["over"], totals[k - 1], info.get("event", {}).get("name")
                             if isinstance(info.get("event"), dict) else None, comp))
                score += dl["runs"]["total"]
                legal += 1 if is_legal else 0
                wkts += sum(1 for w in dl.get("wickets", []) if w["kind"] not in NOT_OUT_KINDS)
    return rows, outs


COLS = ["match_id", "date", "fmt", "innings_no", "team", "opp", "batter", "bowler",
        "runs", "faced", "legal", "dismissed", "kind", "score_before", "wkts_before",
        "legal_before", "target", "max_overs", "winner", "over", "team_total", "event", "comp"]


def load_folder(folder, since_year=2015, men_only=True, limit=None):
    """Read every Cricsheet .json file (or .zip of them) in a folder."""
    all_rows, all_outs, n = [], [], 0

    def handle(raw, mid):
        nonlocal n
        data = json.loads(raw)
        if int(data["info"]["dates"][0][:4]) < since_year:
            return
        r, o = _parse_match(data, mid, men_only)
        all_rows.extend(r)
        all_outs.extend(o)
        n += 1

    for root, _, files in os.walk(folder):
        for f in sorted(files):
            p = os.path.join(root, f)
            if f.endswith(".json"):
                with open(p, "rb") as fh:
                    handle(fh.read(), f[:-5])
            elif f.endswith(".zip"):
                with zipfile.ZipFile(p) as z:
                    for name in z.namelist():
                        if name.endswith(".json"):
                            handle(z.read(name), f[:-4] + "/" + name[:-5])
            if limit and n >= limit:
                break
    balls = pd.DataFrame(all_rows, columns=COLS)
    balls["date"] = pd.to_datetime(balls["date"])
    outs = pd.DataFrame(all_outs, columns=["match_id", "innings_no", "batter", "out_kind"])
    return balls, outs
