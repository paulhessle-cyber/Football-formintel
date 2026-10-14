"""Paper betting: log every flagged bet, then settle it against the result.

A bet is "placed" (1 unit, level stakes) the FIRST time a selection is
flagged as value, at the best price and the average price shown then.
Later runs don't change it - that's what you'd have got if you'd acted.

Once the result is in the data, each bet is settled with:
  * profit at the best price and at the average price
  * closing-line value: bet price vs the de-vigged closing price
    (Betfair Exchange close, bookmaker average close as fallback)

Matches with no result 10 days after kick-off are marked void.
"""
import json

import numpy as np
import pandas as pd

from . import config

PAPER_FILE = config.LOG_DIR / "paper_bets.csv"
KEY = ["date", "div", "home", "away", "selection"]
LABELS = {"h": "Home", "d": "Draw", "a": "Away", "o25": "Over 2.5",
          "u25": "Under 2.5", "ah_home": "AH Home", "ah_away": "AH Away"}


def _load() -> pd.DataFrame:
    if PAPER_FILE.exists():
        return pd.read_csv(PAPER_FILE)
    return pd.DataFrame(columns=KEY)


def record(preds: list[dict], fixtures: pd.DataFrame, run_ts: str):
    """Add new value selections from this run (existing bets untouched)."""
    fx = fixtures.set_index(["div", "home", "away"])
    rows = []
    for p in preds:
        try:
            r = fx.loc[(p["div"], p["home"], p["away"])]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
        except KeyError:
            continue
        for sel in p["value"]:
            if sel.startswith("ah_"):
                side = sel[3:]
                am = p["ah_market"] or {}
                best = am.get(f"{side}_odds")
                avg = r[f"avg_ah{side[0]}"]
                prob, line, edge = None, am.get("line"), am.get(f"{side}_edge")
            else:
                best = p["odds"][sel]
                avg = r[{"h": "avg_h", "d": "avg_d", "a": "avg_a",
                         "o25": "avg_o25", "u25": "avg_u25"}[sel]]
                prob, line, edge = p["prob"][sel], None, p["edge"][sel]
            rows.append({
                "date": p["date"], "time": p["time"], "div": p["div"],
                "home": p["home"], "away": p["away"], "selection": sel,
                "label": LABELS.get(sel, sel), "line": line,
                "odds_best": best,
                "odds_avg": round(float(avg), 2) if pd.notna(avg) else None,
                "prob": prob, "edge": edge,
                "news": bool(p["news"]), "news_note": " | ".join(p["news"]),
                "flagged_at": run_ts, "version": config.VERSION,
                "status": "open",
            })
    if not rows:
        return
    old = _load()
    new = pd.DataFrame(rows)
    allrows = pd.concat([old, new], ignore_index=True)
    allrows = allrows.drop_duplicates(KEY, keep="first")   # first flag wins
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    allrows.to_csv(PAPER_FILE, index=False)


# ---------- settlement ----------

def _ah_profit(hg, ag, line, odds, side):
    """Profit on 1 unit for an Asian handicap (quarter lines split)."""
    margin = hg - ag if side == "home" else ag - hg
    hcap = line if side == "home" else -line
    q = round(hcap * 4)
    parts = [(q - 1) / 4, (q + 1) / 4] if q % 2 else [hcap]
    total = 0.0
    for part in parts:
        x = margin + part
        total += (odds - 1 if x > 0 else 0.0 if x == 0 else -1.0) / len(parts)
    return total


def _close_prob(m: pd.Series, sel: str, line):
    def dv(*cols):
        o = np.array([m.get(c, np.nan) for c in cols], float)
        if np.all(np.isfinite(o) & (o > 1)):
            return (1 / o) / (1 / o).sum()
        return None

    if sel in ("h", "d", "a"):
        i = "hda".index(sel)
        for cols in (("bfec_h", "bfec_d", "bfec_a"), ("avgc_h", "avgc_d", "avgc_a")):
            p = dv(*cols)
            if p is not None:
                return p[i]
    elif sel in ("o25", "u25"):
        i = 0 if sel == "o25" else 1
        for cols in (("bfec_o25", "bfec_u25"), ("avgc_o25", "avgc_u25")):
            p = dv(*cols)
            if p is not None:
                return p[i]
    else:  # AH: only comparable when the closing line equals our line
        cl = m.get("ahc_line", np.nan)
        if line is not None and np.isfinite(cl) and abs(cl - float(line)) < 1e-6:
            p = dv("avgc_ahh", "avgc_aha")
            if p is not None:
                return p[0 if sel == "ah_home" else 1]
    return None


def settle(results: pd.DataFrame):
    """Settle open bets whose result is now in the data."""
    bets = _load()
    if bets.empty:
        return bets
    for c in ("result", "profit_best", "profit_avg", "close_prob", "clv"):
        if c not in bets.columns:
            bets[c] = np.nan
    bets["result"] = bets["result"].astype(object)
    done = results.dropna(subset=["hg", "ag"]).copy()
    done["date_s"] = done["date"].dt.strftime("%Y-%m-%d")
    done = done.drop_duplicates(["date_s", "div", "home", "away"]).set_index(
        ["date_s", "div", "home", "away"])
    today = pd.Timestamp.now().normalize()

    for i, b in bets[bets["status"] == "open"].iterrows():
        key = (b["date"], b["div"], b["home"], b["away"])
        if key not in done.index:
            if pd.Timestamp(b["date"]) < today - pd.Timedelta(days=10):
                bets.loc[i, ["status", "profit_best", "profit_avg"]] = ["void", 0.0, 0.0]
            continue
        m = done.loc[key]
        hg, ag = int(m["hg"]), int(m["ag"])
        sel = b["selection"]
        line = None if pd.isna(b.get("line")) else float(b["line"])

        def profit(odds):
            if pd.isna(odds):
                return np.nan
            odds = float(odds)
            if sel in ("h", "d", "a"):
                won = {"h": hg > ag, "d": hg == ag, "a": hg < ag}[sel]
                return odds - 1 if won else -1.0
            if sel in ("o25", "u25"):
                won = (hg + ag > 2.5) if sel == "o25" else (hg + ag < 2.5)
                return odds - 1 if won else -1.0
            return _ah_profit(hg, ag, line, odds, sel[3:])

        cp = _close_prob(m, sel, line)
        bets.loc[i, "status"] = "settled"
        bets.loc[i, "result"] = f"{hg}-{ag}"
        bets.loc[i, "profit_best"] = round(profit(b["odds_best"]), 3)
        bets.loc[i, "profit_avg"] = round(profit(b["odds_avg"]), 3)
        if cp is not None:
            bets.loc[i, "close_prob"] = round(float(cp), 4)
            bets.loc[i, "clv"] = round(float(b["odds_best"]) * float(cp) - 1, 4)
    bets.to_csv(PAPER_FILE, index=False)
    return bets


# ---------- summary for the website ----------

def _stats(b: pd.DataFrame) -> dict:
    s = b[b["status"] == "settled"]
    out = {"bets": int(len(b)), "open": int((b["status"] == "open").sum()),
           "settled": int(len(s))}
    if len(s):
        clv = s["clv"].dropna()
        out.update(
            profit_best=round(float(s["profit_best"].sum()), 2),
            roi_best=round(float(s["profit_best"].mean()), 4),
            profit_avg=round(float(s["profit_avg"].dropna().sum()), 2),
            roi_avg=round(float(s["profit_avg"].dropna().mean()), 4) if s["profit_avg"].notna().any() else None,
            avg_clv=round(float(clv.mean()), 4) if len(clv) else None,
            beat_close_pct=round(float((clv > 0).mean()), 3) if len(clv) else None,
            win_pct=round(float((s["profit_best"] > 0).mean()), 3),
            avg_odds=round(float(s["odds_best"].mean()), 2))
    return out


def write_summary(bets: pd.DataFrame, run_ts: str):
    out = {"generated": run_ts, "overall": {"bets": 0}, "by_type": {}, "bets": []}
    if not bets.empty:
        bets = bets.copy()
        bets["news"] = bets["news"].astype(str).str.lower() == "true"
        out["overall"] = _stats(bets)
        out["by_type"] = {
            "news": _stats(bets[bets["news"]]),
            "no_news": _stats(bets[~bets["news"]]),
            "1x2": _stats(bets[bets["selection"].isin(["h", "d", "a"])]),
            "ou25": _stats(bets[bets["selection"].isin(["o25", "u25"])]),
            "ah": _stats(bets[bets["selection"].str.startswith("ah_")]),
        }
        # running profit (settled, in kick-off order) for the chart
        s = bets[bets["status"] == "settled"].sort_values(["date", "time"])
        out["curve"] = [round(float(x), 2) for x in s["profit_best"].cumsum()]
        recent = bets.sort_values(["date", "time"], ascending=False).head(300)
        cols = ["date", "time", "div", "home", "away", "label", "line", "odds_best",
                "odds_avg", "edge", "news", "status", "result", "profit_best", "clv"]
        recent = recent[[c for c in cols if c in recent.columns]]
        out["bets"] = json.loads(recent.to_json(orient="records"))
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DATA_DIR / "paper.json").write_text(json.dumps(out, indent=1))
