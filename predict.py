"""Daily run: refresh data, refit, predict upcoming fixtures.

Writes docs/data/predictions.json for the website and appends every
prediction to logs/predictions_log.csv BEFORE kick-off, so results can be
scored honestly later.

Usage:  python predict.py [--no-download]
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fp import blend, config, data, markets, model, news


def num(x, nd=3):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _shift(p_blend: np.ndarray, p_news: np.ndarray, p_base: np.ndarray) -> np.ndarray:
    """Apply the news effect (news model / base model) at full strength."""
    q = p_blend * np.clip(p_news, 1e-9, 1) / np.clip(p_base, 1e-9, 1)
    return q / q.sum()


def predict_fixture(tm: model.TeamModel, r: pd.Series, events: pd.DataFrame,
                    weights: dict) -> dict:
    lam0, mu0 = tm.rates(r["home"], r["away"], r["div"])
    hm, am, notes = news.multipliers(events, r["home"], r["away"], r["date"])
    lam, mu_ = lam0 * hm, mu0 * am

    # 1. team model, before and after news
    m0 = tm.score_matrix(lam0, mu0)
    mn = tm.score_matrix(lam, mu_)
    base1, news1 = np.array(markets.one_x_two(m0)), np.array(markets.one_x_two(mn))
    base2, news2 = np.array(markets.over_under(m0)), np.array(markets.over_under(mn))

    # 2. blend the pre-news model with the market, using backtested weights
    k1 = blend.devig_rows([[r["avg_h"], r["avg_d"], r["avg_a"]]])
    k2 = blend.devig_rows([[r["avg_o25"], r["avg_u25"]]])
    b1 = blend.pool(base1[None], k1, weights["w_1x2"])[0]
    b2 = blend.pool(base2[None], k2, weights["w_ou25"])[0]

    # 3. news moves the blended view by the same amount it moves the model
    f1, f2 = _shift(b1, news1, base1), _shift(b2, news2, base2)

    # 4. scoreline matrix consistent with the final 1X2 and O/U (for AH)
    m = blend.fit_matrix(mn, f1, f2)
    ph, pd_, pa = f1
    po, pu = f2
    ah_line, ah_h, ah_a = markets.ah_fair_line(m)

    def best(mx, av):
        return r[mx] if np.isfinite(r[mx]) else r[av]

    odds = {"h": best("max_h", "avg_h"), "d": best("max_d", "avg_d"),
            "a": best("max_a", "avg_a"), "o25": best("max_o25", "avg_o25"),
            "u25": best("max_u25", "avg_u25")}
    probs = {"h": ph, "d": pd_, "a": pa, "o25": po, "u25": pu}
    edges = {k: num(markets.edge(probs[k], odds[k])) for k in probs}
    has_mkt = bool(np.all(np.isfinite(k1)))
    mkt = list(k1[0]) if has_mkt else None

    ah_mkt = None
    if np.isfinite(r["ah_line"]):
        ahh = best("max_ahh", "avg_ahh")
        aha = best("max_aha", "avg_aha")
        ah_mkt = {"line": num(r["ah_line"], 2),
                  "home_odds": num(ahh, 2), "away_odds": num(aha, 2),
                  "home_edge": num(markets.ah_ev(m, r["ah_line"], ahh)) if np.isfinite(ahh) else None,
                  "away_edge": num(markets.ah_ev(m, r["ah_line"], aha, "away")) if np.isfinite(aha) else None}

    value = [k for k, e in edges.items() if e is not None and e >= config.MIN_EDGE]
    if ah_mkt:
        for side in ("home", "away"):
            e = ah_mkt[f"{side}_edge"]
            if e is not None and e >= config.MIN_EDGE:
                value.append(f"ah_{side}")

    return {
        "div": r["div"], "league": config.LEAGUES.get(r["div"], r["div"]),
        "date": r["date"].strftime("%Y-%m-%d"), "time": str(r["time"]) if pd.notna(r["time"]) else "",
        "home": r["home"], "away": r["away"],
        "xg": {"home": num(float(np.arange(m.shape[0]) @ m.sum(axis=1)), 2),
               "away": num(float(np.arange(m.shape[1]) @ m.sum(axis=0)), 2)},
        "xg_model": {"home": num(lam, 2), "away": num(mu_, 2)},
        "xg_base": {"home": num(lam0, 2), "away": num(mu0, 2)},
        "prob": {k: num(v) for k, v in probs.items()},
        "prob_base": {"h": num(base1[0]), "d": num(base1[1]), "a": num(base1[2])},
        "prob_model": {"h": num(news1[0]), "d": num(news1[1]), "a": num(news1[2]),
                       "o25": num(news2[0]), "u25": num(news2[1])},
        "no_market": not has_mkt,
        "weights": {"w_1x2": weights["w_1x2"], "w_ou25": weights["w_ou25"]},
        "prob_market": None if mkt is None else {"h": num(mkt[0]), "d": num(mkt[1]), "a": num(mkt[2])},
        "fair": {k: markets.fair(v) for k, v in probs.items()},
        "odds": {k: num(v, 2) for k, v in odds.items()},
        "edge": edges,
        "ah_fair": {"line": ah_line, "home": num(ah_h, 2), "away": num(ah_a, 2)},
        "ah_market": ah_mkt,
        "news": notes,
        "value": value,
        "unknown_team": [t for t in (r["home"], r["away"]) if t not in tm.attack],
    }


def ratings_table(tm: model.TeamModel) -> list[dict]:
    rows = []
    for t, lg in tm.team_league.items():
        if lg not in config.LEAGUES:
            continue
        rows.append({"team": t, "div": lg,
                     "attack": round(tm.attack[t], 3),
                     "defence": round(tm.defence[t], 3),
                     "overall": round(tm.attack[t] + tm.defence[t], 3)})
    return sorted(rows, key=lambda x: -x["overall"])


def append_log(preds: list[dict], run_ts: str):
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.LOG_DIR / "predictions_log.csv"
    rows = [{"run_ts": run_ts, "version": config.VERSION, "div": p["div"],
             "date": p["date"], "time": p["time"], "home": p["home"], "away": p["away"],
             "xg_h": p["xg"]["home"], "xg_a": p["xg"]["away"],
             "p_h": p["prob"]["h"], "p_d": p["prob"]["d"], "p_a": p["prob"]["a"],
             "p_o25": p["prob"]["o25"],
             "base_p_h": p["prob_base"]["h"], "base_p_d": p["prob_base"]["d"],
             "base_p_a": p["prob_base"]["a"],
             "model_p_h": p["prob_model"]["h"], "model_p_d": p["prob_model"]["d"],
             "model_p_a": p["prob_model"]["a"],
             "mkt_p_h": (p["prob_market"] or {}).get("h"),
             "mkt_p_d": (p["prob_market"] or {}).get("d"),
             "mkt_p_a": (p["prob_market"] or {}).get("a"),
             "w_1x2": p["weights"]["w_1x2"], "w_ou25": p["weights"]["w_ou25"],
             "odds_h": p["odds"]["h"], "odds_d": p["odds"]["d"], "odds_a": p["odds"]["a"],
             "odds_o25": p["odds"]["o25"], "odds_u25": p["odds"]["u25"],
             "news": " | ".join(p["news"]), "value": ",".join(p["value"])}
            for p in preds]
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        # one row per match per day: keep the latest run of the day
        new["run_day"] = new["run_ts"].str[:10]
        old["run_day"] = old["run_ts"].astype(str).str[:10]
        key = ["run_day", "div", "date", "home", "away"]
        allrows = pd.concat([old, new]).drop_duplicates(key, keep="last")
        allrows = allrows.drop(columns="run_day")
    else:
        allrows = new
    allrows.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    run_ts = datetime.now(timezone.utc).isoformat(timespec="minutes")

    results = data.load_results(refresh=not a.no_download)
    tm = model.fit(data.completed(results))

    error = None
    try:
        if a.no_download and (config.RAW_DIR / "fixtures.csv").exists():
            raw = pd.read_csv(config.RAW_DIR / "fixtures.csv")
            fixtures = data.normalise(raw[raw["Div"].isin(config.LEAGUES)])
        else:
            fixtures = data.download_fixtures()
    except Exception as e:                       # site down etc.
        fixtures, error = pd.DataFrame(), f"Fixtures unavailable: {e}"

    events = news.load()
    weights = blend.load()
    preds = []
    if not fixtures.empty:
        today = pd.Timestamp.now().normalize()
        fixtures = fixtures[fixtures["date"] >= today].sort_values(["date", "time", "div"])
        preds = [predict_fixture(tm, r, events, weights) for _, r in fixtures.iterrows()]

    out = {"version": config.VERSION, "generated": run_ts,
           "model_fitted_to": str(tm.fitted_to.date()),
           "min_edge": config.MIN_EDGE, "error": error,
           "weights": {"w_1x2": weights["w_1x2"], "w_ou25": weights["w_ou25"]},
           "home_advantage": {k: round(float(np.exp(v)), 3) for k, v in tm.home_adv.items()},
           "fixtures": preds, "ratings": ratings_table(tm)}
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DATA_DIR / "predictions.json").write_text(json.dumps(out, indent=1))
    if preds:
        append_log(preds, run_ts)
    print(f"v{config.VERSION}: {len(preds)} fixtures predicted"
          + (f" ({error})" if error else ""))


if __name__ == "__main__":
    main()
