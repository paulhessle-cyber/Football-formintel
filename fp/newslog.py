"""News scorecard: did the betting market later move the way our tags said?

Value bets are rare, so judging news tags by paper bets alone would take
years. Instead every tagged MATCH is scored:

  shift = how far the tags moved our view of the match
  move  = how far the bookmakers' price moved between tagging and kick-off

Both are measured on "home win % minus away win %". If tags carry real
information the market hasn't priced, `move` follows `shift`:

  * moved our way  - share of tagged matches where the price moved in the
                     direction the tags predicted
  * capture        - how much of the predicted shift the market later made
                     (1.0 = all of it, 0 = none: already priced in or wrong,
                     below 0 = it moved the other way)

A snapshot is taken the first time a match appears with tags and is not
changed afterwards.
"""
import json

import numpy as np
import pandas as pd

from . import blend, config

NEWS_LOG = config.LOG_DIR / "news_log.csv"
NEWS_RUNS = config.DATA_DIR / "news_runs.csv"     # one row per run of the news task
KEY = ["date", "div", "home", "away"]


def _load() -> pd.DataFrame:
    if NEWS_LOG.exists():
        return pd.read_csv(NEWS_LOG)
    return pd.DataFrame(columns=KEY)


def record(preds: list[dict], run_ts: str):
    rows = []
    for p in preds:
        if not p["news"] or not p.get("prob_market"):
            continue
        m, f = p["prob_market"], p["prob"]
        rows.append({
            "date": p["date"], "time": p["time"], "div": p["div"],
            "home": p["home"], "away": p["away"], "tagged_at": run_ts,
            "mkt_h": m["h"], "mkt_d": m["d"], "mkt_a": m["a"],
            "our_h": f["h"], "our_d": f["d"], "our_a": f["a"],
            "tags": " | ".join(p["news"]), "status": "open",
        })
    if not rows:
        return
    allrows = pd.concat([_load(), pd.DataFrame(rows)], ignore_index=True)
    allrows = allrows.drop_duplicates(KEY, keep="first")
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    allrows.to_csv(NEWS_LOG, index=False)


def settle(results: pd.DataFrame) -> pd.DataFrame:
    log = _load()
    if log.empty:
        return log
    for c in ("result", "close_h", "close_d", "close_a", "shift", "move"):
        if c not in log.columns:
            log[c] = np.nan
    log["result"] = log["result"].astype(object)
    done = results.dropna(subset=["hg", "ag"]).copy()
    done["date_s"] = done["date"].dt.strftime("%Y-%m-%d")
    done = done.drop_duplicates(["date_s", "div", "home", "away"]).set_index(
        ["date_s", "div", "home", "away"])
    today = pd.Timestamp.now().normalize()

    for i, b in log[log["status"] == "open"].iterrows():
        key = (b["date"], b["div"], b["home"], b["away"])
        if key not in done.index:
            if pd.Timestamp(b["date"]) < today - pd.Timedelta(days=10):
                log.loc[i, "status"] = "void"
            continue
        m = done.loc[key]
        close = blend.devig_rows([[m.get("bfec_h"), m.get("bfec_d"), m.get("bfec_a")]])[0]
        if not np.all(np.isfinite(close)):
            close = blend.devig_rows([[m.get("avgc_h"), m.get("avgc_d"), m.get("avgc_a")]])[0]
        log.loc[i, "result"] = f"{int(m['hg'])}-{int(m['ag'])}"
        log.loc[i, "status"] = "settled"
        if np.all(np.isfinite(close)):
            log.loc[i, ["close_h", "close_d", "close_a"]] = np.round(close, 4)
            sup = lambda h, a: float(h) - float(a)          # home% minus away%
            base = sup(b["mkt_h"], b["mkt_a"])
            log.loc[i, "shift"] = round(sup(b["our_h"], b["our_a"]) - base, 4)
            log.loc[i, "move"] = round(sup(close[0], close[2]) - base, 4)
    log.to_csv(NEWS_LOG, index=False)
    return log


def write_summary(log: pd.DataFrame, run_ts: str):
    out = {"generated": run_ts, "tagged": 0, "settled": 0, "matches": [], "last_run": None}
    try:                                   # when did the news task last report in?
        runs = pd.read_csv(NEWS_RUNS)
        if len(runs):
            last = runs.iloc[-1]
            out["last_run"] = {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                               for k, v in last.items()}
            out["runs"] = int(len(runs))
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        pass
    if not log.empty:
        out["tagged"] = int(len(log))
        s = log[(log["status"] == "settled") & log.get("shift", pd.Series(dtype=float)).notna()]
        s = s[s["shift"].abs() >= 0.002]
        out["settled"] = int(len(s))
        if len(s):
            out["moved_our_way_pct"] = round(float((np.sign(s["move"]) == np.sign(s["shift"])).mean()), 3)
            out["capture"] = round(float((s["move"] * s["shift"]).sum() / (s["shift"] ** 2).sum()), 2)
            out["avg_shift_pts"] = round(float(s["shift"].abs().mean() * 100), 1)
        recent = log.sort_values(["date", "time"], ascending=False).head(60)
        cols = ["date", "home", "away", "tags", "status", "result", "shift", "move"]
        out["matches"] = json.loads(recent[[c for c in cols if c in recent.columns]].to_json(orient="records"))
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DATA_DIR / "news.json").write_text(json.dumps(out, indent=1))
