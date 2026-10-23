"""Check every URL in data/disease_info.json (run from a machine with internet access).

    python scripts/check_links.py
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

data = json.loads((Path(__file__).resolve().parent.parent / "data" / "disease_info.json").read_text("utf-8"))
urls = {s["url"] for s in data.get("general_sources", [])}
for entry in data["classes"].values():
    urls |= {s["url"] for s in entry.get("sources", [])}

bad = 0
for url in sorted(urls):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (link check)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"OK   {r.status}  {url}")
    except urllib.error.HTTPError as e:
        bad += e.code >= 400 and e.code != 403   # some sites block scripts with 403; open those by hand
        print(f"{'WARN' if e.code == 403 else 'FAIL'} {e.code}  {url}")
    except Exception as e:  # noqa: BLE001
        bad += 1
        print(f"FAIL ---  {url}  ({e})")
sys.exit(1 if bad else 0)
