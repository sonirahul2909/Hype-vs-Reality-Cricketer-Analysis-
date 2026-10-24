"""Stage 3: hype data.

 * Attention = monthly Wikipedia pageviews (free, no key, history back to 2015).
 * Mood      = VADER sentiment of recent GDELT headlines.
   NOTE: GDELT's DOC API only searches roughly the last 3 months, so mood is a
   *recent snapshot*, not a time series. Everything is cached on disk.
"""
import json
import os
import re
import time
from urllib.parse import quote

import pandas as pd
import requests

UA = {"User-Agent": "HypeVsRealityCricket/1.0 (student project; "
                    + os.environ.get("HVR_CONTACT", "no-contact-given") + ")"}
LAST_ERROR = ""  # why the most recent request failed (printed by fetch_hype.py)


def _cache_get(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _cache_put(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def _get(url, params=None, tries=3, wait=2.0):
    global LAST_ERROR
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=20)
            if r.status_code == 404:
                LAST_ERROR = "404 not found"
                return None
            if r.status_code == 429:
                LAST_ERROR = "429 rate limited"
                time.sleep(max(wait, 5) * (i + 1))
                continue
            r.raise_for_status()
            LAST_ERROR = ""
            return r
        except requests.RequestException as e:
            LAST_ERROR = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(wait * (i + 1))
    return None


def load_overrides(path):
    try:
        t = pd.read_csv(path)
        return dict(zip(t["cricsheet_name"], t["wikipedia_title"]))
    except FileNotFoundError:
        return {}


def resolve_title(name, cache_dir, overrides):
    """Cricsheet name ('V Kohli') -> Wikipedia article title ('Virat Kohli')."""
    if name in overrides:
        return overrides[name]
    p = os.path.join(cache_dir, "titles.json")
    cache = _cache_get(p) or {}
    if cache.get(name):  # earlier misses (None) are retried
        return cache[name]
    # Cricsheet names are 'initials + surname' ("AK Markram"), Wikipedia uses full names
    # ("Aiden Markram"). So: try the full string first, then the surname alone, and for the
    # surname search require the first letter of the first name to match the initials.
    parts = name.split()
    surname = parts[-1].lower()
    first_initial = parts[0][0].lower() if len(parts) > 1 else None
    title, any_ok = None, False
    for query, check_initial in [(f"{name} cricketer", False), (f"{parts[-1]} cricketer", True)]:
        r = _get("https://en.wikipedia.org/w/api.php",
                 {"action": "query", "generator": "search", "gsrsearch": query,
                  "gsrlimit": 10, "prop": "description", "format": "json"}, wait=8)
        if r is None:
            continue
        any_ok = True
        pages = sorted(r.json().get("query", {}).get("pages", {}).values(),
                       key=lambda pg: pg.get("index", 99))
        for pg in pages:
            t = pg["title"].lower()
            if "cricket" not in pg.get("description", "").lower() or surname not in t:
                continue
            if check_initial and first_initial and not t.startswith(first_initial):
                continue
            title = pg["title"]
            break
        time.sleep(0.8)
        if title:
            break
    if not any_ok:  # network / rate-limit problem: do NOT cache, so a re-run tries again
        return None
    cache[name] = title
    _cache_put(p, cache)
    return title


def monthly_pageviews(title, start="20150701", end=None, cache_dir="data/cache"):
    """Returns a DataFrame [month, views] or an empty one."""
    end = end or time.strftime("%Y%m01")
    p = os.path.join(cache_dir, "pageviews", re.sub(r"\W+", "_", title) + ".json")
    cached = _cache_get(p)
    if cached is None:
        url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
               f"en.wikipedia/all-access/user/{quote(title.replace(' ', '_'), safe='')}"
               f"/monthly/{start}00/{end}00")
        r = _get(url)
        if r is None:
            return pd.DataFrame(columns=["month", "views"])
        cached = [(i["timestamp"][:6], i["views"]) for i in r.json().get("items", [])]
        _cache_put(p, cached)
        time.sleep(0.2)
    df = pd.DataFrame(cached, columns=["month", "views"])
    df["month"] = pd.to_datetime(df["month"], format="%Y%m")
    return df


def recent_headline_tone(fullname, cache_dir="data/cache"):
    """Mean VADER compound score of recent headlines + how many were found."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    p = os.path.join(cache_dir, "gdelt", re.sub(r"\W+", "_", fullname) + ".json")
    cached = _cache_get(p)
    if cached is None:
        r = _get("https://api.gdeltproject.org/api/v2/doc/doc",
                 {"query": f'"{fullname}" cricket sourcelang:english', "mode": "artlist",
                  "maxrecords": 100, "format": "json", "timespan": "3months"}, wait=6)
        if r is None:
            return None
        try:
            titles = [a["title"] for a in r.json().get("articles", [])]
        except ValueError:
            return None
        an = SentimentIntensityAnalyzer()
        scores = [an.polarity_scores(t)["compound"] for t in titles]
        cached = {"n": len(scores), "tone": sum(scores) / len(scores) if scores else None}
        _cache_put(p, cached)
        time.sleep(5.5)  # GDELT asks for ~1 request / 5 s
    return cached