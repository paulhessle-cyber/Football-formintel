"""Betfair Exchange prices recorded by the odds proxy on the droplet.

The proxy (server/football.js) looks at Betfair's Match Odds and Over/Under
2.5 markets for every League One, League Two and National League match, up
to every 2 minutes in the last hour. Each run of predict.py asks it for the
last three weeks and keeps a compact summary per match in
data/betfair/YYYY-MM.json:

  first      the first price recorded (usually ~3 days out)
  h24 / h6 / h1 / m10   the price 24h, 6h, 1h and 10 min before kick-off
  close      the last price before the match went in-play
  now        the latest price (for matches still to come)

each as fair probabilities (mid of back and lay, margin removed), plus the
money matched and the result. Full price histories stay on the droplet.

Betfair's team names differ from football-data's ("Oxford Utd" v "Oxford"),
so matches are paired by league + kick-off date + closest names.
"""
import difflib
import json
import re

import numpy as np
import pandas as pd
import requests

from . import blend, config

BF_DIR = config.DATA_DIR / "betfair"
POINTS = {"h24": 24 * 60, "h6": 6 * 60, "h1": 60, "m10": 10}      # minutes before kick-off
TOL = {"h24": 120, "h6": 45, "h1": 15, "m10": 6}

ALIASES = {      # Betfair-ish name -> football-data name, where fuzzy matching might slip
    "mk dons": "Milton Keynes Dons", "milton keynes": "Milton Keynes Dons",
    "peterborough": "Peterboro", "bristol rovers": "Bristol Rvs", "sheff wed": "Sheffield Weds",
    "sheffield wednesday": "Sheffield Weds", "dagenham": "Dag and Red", "dag and red": "Dag and Red",
    "dagenham and redbridge": "Dag and Red", "oxford utd": "Oxford", "oxford united": "Oxford",
    "cambridge utd": "Cambridge", "cambridge united": "Cambridge", "boston utd": "Boston Utd",
    "boston united": "Boston Utd", "notts co": "Notts County", "fleetwood": "Fleetwood Town",
    "afc fylde": "Fylde", "fc halifax town": "Halifax", "fc halifax": "Halifax", "halifax town": "Halifax",
    "solihull moors": "Solihull", "harrogate town": "Harrogate", "york city": "York",
    "accrington stanley": "Accrington", "leicester city": "Leicester", "luton town": "Luton",
    "wigan athletic": "Wigan", "plymouth argyle": "Plymouth", "huddersfield town": "Huddersfield",
    "rotherham utd": "Rotherham", "rotherham united": "Rotherham", "stockport county": "Stockport",
    "exeter city": "Exeter", "burton albion": "Burton", "lincoln city": "Lincoln",
    "port vale": "Port Vale", "salford city": "Salford", "swindon town": "Swindon",
    "tranmere rovers": "Tranmere", "southend utd": "Southend", "southend united": "Southend",
    "woking fc": "Woking", "yeovil town": "Yeovil", "truro city": "Truro", "tamworth fc": "Tamworth",
    "forest green rovers": "Forest Green", "crawley": "Crawley Town", "braintree": "Braintree Town",
    "ebbsfleet utd": "Ebbsfleet", "ebbsfleet united": "Ebbsfleet", "wealdstone fc": "Wealdstone",
    "sutton utd": "Sutton", "sutton united": "Sutton", "barrow afc": "Barrow",
    "northampton town": "Northampton", "doncaster rovers": "Doncaster", "bradford city": "Bradford",
    "crewe alexandra": "Crewe", "cheltenham town": "Cheltenham", "shrewsbury town": "Shrewsbury",
    "grimsby town": "Grimsby", "colchester utd": "Colchester", "colchester united": "Colchester",
    "gillingham fc": "Gillingham", "newport co": "Newport County", "scunthorpe utd": "Scunthorpe",
    "scunthorpe united": "Scunthorpe", "hartlepool utd": "Hartlepool", "hartlepool united": "Hartlepool",
    "morecambe fc": "Morecambe", "maidenhead utd": "Maidenhead", "maidenhead united": "Maidenhead",
    "brackley": "Brackley Town", "hornchurch": "Hornchurch", "aldershot town": "Aldershot",
    "altrincham fc": "Altrincham", "boreham wood fc": "Boreham Wood", "eastleigh fc": "Eastleigh",
    "dorking wanderers": "Dorking", "kidderminster harriers": "Kidderminster",
    "gateshead fc": "Gateshead", "rochdale afc": "Rochdale", "oldham athletic": "Oldham",
    "carlisle utd": "Carlisle", "carlisle united": "Carlisle", "walsall fc": "Walsall",
    "barnsley fc": "Barnsley", "blackpool fc": "Blackpool", "reading fc": "Reading",
    "stevenage fc": "Stevenage", "wycombe wanderers": "Wycombe", "leyton orient": "Leyton Orient",
    "chesterfield fc": "Chesterfield", "mansfield town": "Mansfield", "bromley fc": "Bromley",
    "barnet fc": "Barnet", "oxford city": "Oxford City",
}


def _norm(s: str) -> str:
    s = str(s).lower().replace("&", "and")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _sim(bf: str, fd: str) -> float:
    a, b = _norm(bf), _norm(fd)
    if ALIASES.get(a) == fd:
        return 1.0
    if a == b:
        return 1.0
    score = difflib.SequenceMatcher(None, a, b).ratio()
    if a.startswith(b) or b.startswith(a):           # "Wigan Athletic" ~ "Wigan"
        score = max(score, 0.85)
    return score


# ── fetching and summarising ─────────────────────────────────────────────────

def _fair(look: dict, keys: list) -> list | None:
    """Fair probabilities from one look: mid of back/lay, margin removed."""
    mids = []
    for k in keys:
        bl = look.get(k) or [None, None]
        back, lay = bl[0], bl[1] if len(bl) > 1 else None
        if back and lay and lay / back < 1.25:        # a sensible spread
            mids.append(2 / (1 / back + 1 / lay))     # mid in probability terms
        elif back:
            mids.append(back)
        else:
            return None
    p = blend.devig_rows([mids])[0]
    return [round(float(x), 4) for x in p] if np.all(np.isfinite(p)) else None


def _summary(rec: dict, ko: int, keys: list) -> dict:
    looks = sorted(rec.get("looks") or [], key=lambda l: l["at"])
    out = {"looks": len(looks)}
    if not looks:
        return out
    out["first"] = _fair(looks[0], keys)
    out["first_at"] = looks[0]["at"]
    for name, mins in POINTS.items():
        target = ko - mins * 60000
        best = min((l for l in looks if l["at"] < ko), key=lambda l: abs(l["at"] - target), default=None)
        if best and abs(best["at"] - target) <= TOL[name] * 60000:
            out[name] = _fair(best, keys)
    pre = [l for l in looks if l["at"] <= (rec.get("inplayAt") or ko + 4 * 3600000)]
    if pre:
        last = pre[-1]
        out["last"] = _fair(last, keys)
        out["last_at"] = last["at"]
        out["mt"] = last.get("mt")
        if rec.get("inplayAt") and rec["inplayAt"] - last["at"] <= 15 * 60000:
            out["close"] = out["last"]                # within 15 min of going in-play
    if rec.get("won"):
        out["won"] = rec["won"]
    return out


def update(url: str | None = None, days: int = 21) -> int:
    """Fetch recent matches from the proxy and merge them into data/betfair/."""
    url = url or config.BETFAIR_URL
    if not url:
        return 0
    r = requests.get(url, params={"days": days}, timeout=60, headers={"User-Agent": "football-predictor"})
    r.raise_for_status()
    body = r.json()
    BF_DIR.mkdir(parents=True, exist_ok=True)
    by_month: dict[str, dict] = {}
    for m in body.get("matches", []):
        month = m["d"][:7]
        if month not in by_month:
            f = BF_DIR / f"{month}.json"
            by_month[month] = json.loads(f.read_text()) if f.exists() else {}
        mk = m.get("mk") or {}
        row = {"id": m["id"], "div": m["div"], "date": m["d"], "time": m["t"], "ko": m["ko"],
               "bf_home": m["home"], "bf_away": m["away"]}
        if "MATCH_ODDS" in mk:
            row["mo"] = _summary(mk["MATCH_ODDS"], m["ko"], ["h", "d", "a"])
        if "OVER_UNDER_25" in mk:
            row["ou"] = _summary(mk["OVER_UNDER_25"], m["ko"], ["o", "u"])
        by_month[month][m["id"]] = row
    for month, rows in by_month.items():
        (BF_DIR / f"{month}.json").write_text(json.dumps(rows, separators=(",", ":"), sort_keys=True))
    (BF_DIR / "status.json").write_text(json.dumps(body.get("status", {}), indent=1))
    return len(body.get("matches", []))


def load() -> list[dict]:
    rows = []
    for f in sorted(BF_DIR.glob("????-??.json")):
        try:
            rows.extend(json.loads(f.read_text()).values())
        except (json.JSONDecodeError, OSError):
            continue
    return rows


# ── pairing with football-data matches ───────────────────────────────────────

class Index:
    """Find the Betfair record for a football-data match (div, date, home, away)."""

    def __init__(self, rows: list[dict] | None = None):
        self.by_key: dict[tuple, list] = {}
        for r in rows if rows is not None else load():
            self.by_key.setdefault((r["div"], r["date"]), []).append(r)
        self.unmatched: list[str] = []

    def find(self, div: str, date: str, home: str, away: str) -> dict | None:
        cands = self.by_key.get((div, date), [])
        best, score = None, 0.0
        for r in cands:
            s = min(_sim(r["bf_home"], home), _sim(r["bf_away"], away))
            if s > score:
                best, score = r, s
        if best is not None and score >= 0.6:
            return best
        if cands:
            self.unmatched.append(f"{date} {div} {home} v {away}")
        return None


def match_view(rec: dict | None) -> dict | None:
    """What the website shows for a fixture."""
    if not rec or "mo" not in rec:
        return None
    mo, ou = rec["mo"], rec.get("ou", {})
    now = mo.get("last")
    if not now:
        return None
    first = mo.get("first")
    move = round((now[0] - now[2]) - (first[0] - first[2]), 4) if first else None
    return {"now": now, "first": first, "first_at": mo.get("first_at"), "at": mo.get("last_at"),
            "move": move, "mt": mo.get("mt"), "over": (ou.get("last") or [None])[0]}


def close_probs(rec: dict | None, market: str) -> list | None:
    """Betfair closing fair probabilities ('mo' = home/draw/away, 'ou' = over/under)."""
    if not rec or market not in rec:
        return None
    return rec[market].get("close")
