# Hype vs Reality: Cricket Player Analyzer

Does a player's public hype match what he actually does under pressure?
Type a player, pick a format and time window, and the app scores his innings by match pressure,
measures public attention, and gives a verdict: **Overhyped, Fairly Rated, or Underrated**.

## Quick start (real data)

```bash
pip install -r requirements.txt
python download_data.py          # Cricsheet: Tests, ODIs, T20Is, IPL (~1-2 min)
python build_data.py --no-gdelt  # parse + score + Wikipedia pageviews (several minutes)
streamlit run app.py
```

Optional extras:
- `python build_data.py` (without `--no-gdelt`) also fetches recent headline mood. GDELT asks for about one request
  per 5 seconds, so for 200 players this adds roughly 20 minutes. Results are cached in `data/cache/`.
- `python build_data.py --skip-hype` builds impact scores only (no internet needed after the download).
- Open `data/processed/wiki_titles_used.csv` and check every player matched the right Wikipedia article.
  Fix wrong or missing ones in `data/wiki_overrides.csv` and re-run (cached data makes re-runs fast).
- Open `data/processed/unmapped_bowlers.csv` and add the most common bowlers to `data/bowler_types.csv`
  (Pace / Off-spin / Leg-spin / Left-arm spin). Names must match Cricsheet spelling exactly.

## Demo mode (no internet, fictional players)

```bash
python make_demo_data.py && python build_data.py --demo
DATA_DIR=data_demo/processed streamlit run app.py
```

## How it works

| Stage | File | What it does |
| --- | --- | --- |
| 0 | `cricket/parser.py` | Cricsheet JSON -> one row per ball, with score, wickets, target before the ball |
| 1 | `cricket/scoring.py` | Pressure per ball, pressure-weighted runs, innings impact, clutch table text |
| 2 | `cricket/profile.py` | Phase / bowler type / dismissal splits, strengths and weaknesses, two proxy metrics |
| 3 | `cricket/hype.py` | Wikipedia monthly pageviews, GDELT headlines + VADER sentiment, all cached |
| 4 | `cricket/verdict.py` | Percentiles within a cohort, gap = hype pct - impact pct, verdict + summary text |
| 5 | `app.py` | Streamlit dashboard + Hidden Gems leaderboard |

**Pressure formula.** `P = wicket_factor x rate_factor` (clipped to 0.5-3.0).
`wicket_factor = 1 + 0.15 x (wickets down - 6 x innings progress)` for limited overs
(`1 + 0.10 x (wickets down - 2)` for Tests); `rate_factor = required rate / par rate` when chasing.
Impact of an innings = sum of (runs on each ball x P of that ball).

**Verdict.** `gap = hype percentile - impact percentile`; gap >= +20 Overhyped, <= -20 Underrated.
Percentiles are computed among the tracked players for the same format and window.

## Fairness rules (v2)

- **Competition.** Every match is tagged International or League (IPL etc.) from Cricsheet's `team_type` (falling back to the event name). The app has a Competition dropdown, and international and league cricket are ranked separately. `build_data.py` prints the innings count per format/competition so you can check the split.
- **Roles.** `cricket/roles.py` classifies each player from the last 3 years as Batter, All-rounder or Bowler (where he bats, how much he bowls). Impact is ranked only within the same role group, and **Bowlers get no verdict** because the model scores batting only.
- After pulling these changes run `python build_data.py --skip-hype` once (the hype data is kept) and restart the app.

## Honest limits (also shown in the app)

- Batting only: bowling impact is not modelled. Hype (Wikipedia pageviews) cannot be split by competition.
- Technique and temperament are **proxies** built from scoring patterns, not measurements.
- "Overhyped" has no ground truth. The verdict is a transparent rule, not a learned label.
- Pageviews measure attention (which can be negative), not approval. GDELT's search API only covers about
  the last 3 months, so headline mood is a recent snapshot, not a time series. Google Trends is not used
  (its numbers are not comparable across players).
- Small samples are noisy; the app warns when innings counts are low.
- Check the licence of every data source (Cricsheet, Wikimedia, GDELT) before any commercial use.

## Deploy free on Streamlit Community Cloud

Push this folder to GitHub **including `data/processed/`** (but not `data/raw/` or `data/cache/`,
which `.gitignore` already excludes), then create the app from `app.py`.

## What was tested

The parser, scoring, verdict logic and app were tested end to end on the synthetic demo data set.
The live downloads (Cricsheet, Wikipedia, GDELT) could not be reached from the build environment, so
`download_data.py` and `cricket/hype.py` are written to the documented APIs but have **not** been run against
the live services. If something fails, the scripts print which player or step failed.
