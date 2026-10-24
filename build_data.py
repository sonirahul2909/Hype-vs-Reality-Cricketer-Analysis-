"""Build every table the app needs.  Run after download_data.py:

    python build_data.py                       # real data, fetches hype from the web
    python build_data.py --skip-hype           # impact only (fast; no internet needed)
    python build_data.py --demo                # use the synthetic demo data set
"""
import argparse
import json
import os
import time

import pandas as pd

from cricket import hype as H
from cricket.parser import load_folder
from cricket.profile import attach_bowler_type, load_bowler_types
from cricket.roles import classify_roles
from cricket.scoring import add_pressure, add_phase, innings_table

ap = argparse.ArgumentParser()
ap.add_argument("--raw", default="data/raw")
ap.add_argument("--out", default="data/processed")
ap.add_argument("--since-year", type=int, default=2018)
ap.add_argument("--top-players", type=int, default=200)
ap.add_argument("--min-innings", type=int, default=15, help="min innings in last 3 yrs to be tracked")
ap.add_argument("--bowler-types", default="data/bowler_types.csv")
ap.add_argument("--wiki-overrides", default="data/wiki_overrides.csv")
ap.add_argument("--skip-hype", action="store_true")
ap.add_argument("--no-gdelt", action="store_true", help="skip the headline-mood step")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--limit", type=int, default=None, help="only read N match files (testing)")
a = ap.parse_args()

if a.demo:
    a.raw, a.out = "data_demo/raw", "data_demo/processed"
    a.bowler_types = "data_demo/bowler_types.csv"
os.makedirs(a.out, exist_ok=True)
t0 = time.time()

print("1/5 Parsing matches ...")
balls, outs = load_folder(a.raw, since_year=a.since_year, limit=a.limit)
print(f"    {len(balls):,} deliveries from {balls['match_id'].nunique():,} matches")

print("2/5 Scoring pressure ...")
balls = add_phase(add_pressure(balls))
inn = innings_table(balls, outs)
latest = inn["date"].max()

recent = inn[inn["date"] >= latest - pd.DateOffset(years=3)]
cnt = recent.groupby("batter").size()
tracked = cnt[cnt >= a.min_innings].sort_values(ascending=False).head(a.top_players).index
print(f"    tracking {len(tracked)} players with >= {a.min_innings} innings in the last 3 years")
roles = classify_roles(balls, inn, tracked, latest)
roles.to_parquet(os.path.join(a.out, "roles.parquet"), index=False)
print("    roles:", roles["role_group"].value_counts().to_dict())
print("    innings by format / competition (check this looks right):")
print(inn.groupby(["fmt", "comp"]).size().to_string())
balls = balls[balls["batter"].isin(tracked)].copy()
inn = inn[inn["batter"].isin(tracked)].copy()

print("3/5 Bowler types ...")
lookup = load_bowler_types(a.bowler_types)
balls = attach_bowler_type(balls, lookup)
unm = (balls[balls["bowler_type"] == "Unknown"].groupby("bowler").size()
       .sort_values(ascending=False).head(60).rename("balls_bowled_to_tracked").reset_index())
unm.to_csv(os.path.join(a.out, "unmapped_bowlers.csv"), index=False)
print(f"    {100 * (balls['bowler_type'] != 'Unknown').mean():.0f}% of balls have a known bowler type "
      f"(add names from unmapped_bowlers.csv to {a.bowler_types} to raise this)")

keep = ["match_id", "date", "fmt", "innings_no", "batter", "bowler", "bowler_type", "bowler_kind",
        "runs", "faced", "dismissed", "kind", "pressure", "w_runs", "phase", "over", "wkts_before", "comp"]
balls[keep].to_parquet(os.path.join(a.out, "balls.parquet"), index=False)
inn.to_parquet(os.path.join(a.out, "innings.parquet"), index=False)

print("4/5 Hype data ...")
hype_rows, tones, titles = [], {}, {}
if a.demo:
    hype_rows = pd.read_csv("data_demo/hype_monthly.csv", parse_dates=["month"])
    hype_rows.to_parquet(os.path.join(a.out, "hype_monthly.parquet"), index=False)
    with open(os.path.join(a.out, "hype_tone.json"), "w") as f:
        json.dump({}, f)
elif a.skip_hype:
    print("    skipped (the app will show impact only)")
else:
    ov = H.load_overrides(a.wiki_overrides)
    cache = "data/cache"
    for i, p in enumerate(tracked, 1):
        title = H.resolve_title(p, cache, ov)
        titles[p] = title
        if not title:
            print(f"    [{i}/{len(tracked)}] {p}: no Wikipedia page found - add it to {a.wiki_overrides}")
            continue
        pv = H.monthly_pageviews(title, cache_dir=cache)
        if pv.empty:
            print(f"    [{i}/{len(tracked)}] {p} -> {title}: no pageview data")
            continue
        pv["player"], pv["wiki_title"] = p, title
        hype_rows.append(pv)
        if not a.no_gdelt:
            t = H.recent_headline_tone(title, cache)
            if t:
                tones[p] = t
        if i % 10 == 0:
            print(f"    {i}/{len(tracked)} players done")
    if hype_rows:
        pd.concat(hype_rows).to_parquet(os.path.join(a.out, "hype_monthly.parquet"), index=False)
    with open(os.path.join(a.out, "hype_tone.json"), "w") as f:
        json.dump(tones, f)
    pd.Series(titles, name="wikipedia_title").rename_axis("cricsheet_name").to_csv(
        os.path.join(a.out, "wiki_titles_used.csv"))
    print("    Check wiki_titles_used.csv: a wrong article match means wrong hype for that player.")

print("5/5 Saving metadata ...")
with open(os.path.join(a.out, "meta.json"), "w") as f:
    json.dump({"latest_match": str(latest.date()), "players": int(len(tracked)),
               "matches": int(balls["match_id"].nunique()), "demo": bool(a.demo),
               "built": time.strftime("%Y-%m-%d %H:%M")}, f)
print(f"Done in {time.time() - t0:.0f}s -> {a.out}")
