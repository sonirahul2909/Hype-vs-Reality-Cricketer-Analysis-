"""Download Cricsheet ball-by-ball data (free). Run once:  python download_data.py

Cricsheet data is shared under the Open Data Commons Attribution licence (check
https://cricsheet.org/ for current terms, especially before commercial use).
"""
import io
import os
import sys
import zipfile

import requests

BASE = "https://cricsheet.org/downloads/"
FILES = {"tests": "tests_json.zip", "odis": "odis_json.zip",
         "t20is": "t20s_json.zip", "ipl": "ipl_json.zip"}


def main(out="data/raw"):
    for name, fn in FILES.items():
        dest = os.path.join(out, name)
        if os.path.isdir(dest) and os.listdir(dest):
            print(f"{name}: already downloaded, skipping")
            continue
        print(f"Downloading {fn} ...")
        r = requests.get(BASE + fn, timeout=300,
                         headers={"User-Agent": "hype-vs-reality-cricket/1.0"})
        r.raise_for_status()
        os.makedirs(dest, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(dest)
        print(f"  -> {len(os.listdir(dest))} files in {dest}")


if __name__ == "__main__":
    main(*sys.argv[1:])
