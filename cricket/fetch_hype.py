"""Redo ONLY the hype step (fast; does not re-parse the matches).

    python fetch_hype.py --no-gdelt

Needs data/processed/innings.parquet from build_data.py. It tells you exactly which players
failed and why, and writes data/processed/wiki_titles_used.csv so you can check the matches.
Optional: set HVR_CONTACT="you@example.com" so Wikimedia can identify your requests.
"""
import argparse
import json
import os

import pandas as pd

from cricket import hype as H

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/processed")
ap.add_argument("--wiki-overrides", default="data/wiki_overrides.csv")
ap.add_argument("--no-gdelt", action="store_true")
ap.add_argument("--only", nargs="*", help="only these players (e.g. --only 'AK Markram')")
a = ap.parse_args()

inn = pd.read_parquet(os.path.join(a.out, "innings.parquet"))
tracked = list(inn.groupby("batter").size().sort_values(ascending=False).index)
if a.only:
    tracked = [p for p in tracked if p in a.only]
ov = H.load_overrides(a.wiki_overrides)
cache = "data/cache"
rows, tones, titles, problems = [], {}, {}, []

# quick connectivity check so a network/blocked-request problem is obvious right away
t = H.resolve_title(tracked[0], cache, ov)
if t is None and H.LAST_ERROR:
    print(f"\nCannot reach Wikipedia: {H.LAST_ERROR}\n"
          "Check your internet / VPN / proxy, then run again.")
    raise SystemExit(1)

for i, p in enumerate(tracked, 1):
    title = H.resolve_title(p, cache, ov)
    titles[p] = title
    if not title:
        problems.append((p, "no Wikipedia article matched" + (f" ({H.LAST_ERROR})" if H.LAST_ERROR else "")))
        continue
    pv = H.monthly_pageviews(title, cache_dir=cache)
    if pv.empty:
        problems.append((p, f"no pageviews for '{title}'" + (f" ({H.LAST_ERROR})" if H.LAST_ERROR else "")))
        continue
    pv["player"], pv["wiki_title"] = p, title
    rows.append(pv)
    if not a.no_gdelt:
        tn = H.recent_headline_tone(title, cache)
        if tn:
            tones[p] = tn
    if i % 10 == 0:
        print(f"  {i}/{len(tracked)} players, {len(rows)} with hype data so far")

if rows:
    out = pd.concat(rows)
    if a.only and os.path.exists(os.path.join(a.out, "hype_monthly.parquet")):  # merge with earlier run
        old = pd.read_parquet(os.path.join(a.out, "hype_monthly.parquet"))
        out = pd.concat([old[~old["player"].isin(a.only)], out])
    out.to_parquet(os.path.join(a.out, "hype_monthly.parquet"), index=False)
    with open(os.path.join(a.out, "hype_tone.json"), "w") as f:
        json.dump(tones, f)
pd.Series(titles, name="wikipedia_title").rename_axis("cricsheet_name").to_csv(
    os.path.join(a.out, "wiki_titles_used.csv"))

print(f"\nHype data for {len(rows)} of {len(tracked)} players.")
for p, why in problems[:25]:
    print(f"  - {p}: {why}")
if problems:
    print("Fix problems by adding 'cricsheet_name,wikipedia_title' lines to data/wiki_overrides.csv, "
          "then re-run (use --only to redo just those players).")
print("Now restart the app: streamlit run app.py")