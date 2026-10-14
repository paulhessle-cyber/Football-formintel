"""Walk-forward backtest of the baseline model against closing odds.

Every week the model is refitted on matches played before that week, then
predicts that week's matches, exactly as it would have live. We measure:

  * log loss / Brier vs the de-vigged Pinnacle closing line (the sharpest
    benchmark there is - beating it is the real test)
  * value bets: level stakes on edges >= MIN_EDGE at the pre-match best
    price, with ROI and closing-line value (CLV)

Usage:  python backtest.py [--from 2425] [--no-download]
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fp import config, data, markets, model


def log_loss(p, outcome_idx):
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    return -np.log(p[np.arange(len(p)), outcome_idx]).mean()


def brier(p, outcome_idx):
    p = np.asarray(p, float)
    y = np.zeros_like(p)
    y[np.arange(len(p)), outcome_idx] = 1
    return ((p - y) ** 2).sum(axis=1).mean()


def run(matches: pd.DataFrame, start_season: str) -> pd.DataFrame:
    # train on every league, but only score the leagues we actually predict
    test = matches[(matches["season"] >= start_season) &
                   (matches["div"].isin(config.LEAGUES))]
    start = test["date"].min().normalize()
    start -= pd.Timedelta(days=start.weekday())          # back to Monday
    end = test["date"].max()
    rows, warm = [], None
    week = start
    while week <= end:
        nxt = week + pd.Timedelta(days=7)
        batch = test[(test["date"] >= week) & (test["date"] < nxt)]
        if len(batch):
            try:
                warm = model.fit(matches, as_of=week, warm=warm)
            except ValueError:
                week = nxt
                continue
            for _, r in batch.iterrows():
                lam, mu_ = warm.rates(r["home"], r["away"], r["div"])
                m = warm.score_matrix(lam, mu_)
                ph, pd_, pa = markets.one_x_two(m)
                po, pu = markets.over_under(m, 2.5)
                rows.append({**r.to_dict(), "xg_h": lam, "xg_a": mu_,
                             "p_h": ph, "p_d": pd_, "p_a": pa,
                             "p_o25": po, "p_u25": pu,
                             "ah_ev_h": markets.ah_ev(m, r["ahc_line"], r["pc_ahh"])
                             if np.isfinite(r["ahc_line"]) and np.isfinite(r["pc_ahh"]) else np.nan,
                             "ah_ev_a": markets.ah_ev(m, r["ahc_line"], r["pc_aha"], "away")
                             if np.isfinite(r["ahc_line"]) and np.isfinite(r["pc_aha"]) else np.nan})
        week = nxt
    return pd.DataFrame(rows)


def market_probs(df, cols_sets):
    """De-vigged probabilities from the first odds set that is present."""
    out = np.full((len(df), len(cols_sets[0])), np.nan)
    for cols in cols_sets:
        need = np.isnan(out[:, 0])
        if not need.any():
            break
        odds = df[cols].to_numpy(float)
        ok = need & np.all(odds > 1, axis=1)
        inv = 1 / odds[ok]
        out[ok] = inv / inv.sum(axis=1, keepdims=True)
    return out


def summarise(bt: pd.DataFrame) -> dict:
    res = np.select([bt.hg > bt.ag, bt.hg == bt.ag], [0, 1], 2)
    model_p = bt[["p_h", "p_d", "p_a"]].to_numpy()
    mkt_p = market_probs(bt, [["pc_h", "pc_d", "pc_a"], ["avgc_h", "avgc_d", "avgc_a"]])
    has = ~np.isnan(mkt_p[:, 0])

    total = bt.hg + bt.ag
    ou_res = (total < 2.5).astype(int)           # 0 = over, 1 = under
    model_ou = bt[["p_o25", "p_u25"]].to_numpy()
    mkt_ou = market_probs(bt, [["pc_o25", "pc_u25"], ["avgc_o25", "avgc_u25"]])
    has_ou = ~np.isnan(mkt_ou[:, 0])

    # value bets on 1X2 at the pre-match best price
    bets = []
    price_cols = [("max_h", "avg_h"), ("max_d", "avg_d"), ("max_a", "avg_a")]
    for i, r in enumerate(bt.itertuples(index=False)):
        for k, (mx, av) in enumerate(price_cols):
            odds = getattr(r, mx)
            if not np.isfinite(odds):
                odds = getattr(r, av)
            e = markets.edge(model_p[i, k], odds)
            if e is not None and e >= config.MIN_EDGE:
                won = res[i] == k
                clv = odds * mkt_p[i, k] - 1 if has[i] else np.nan
                bets.append({"div": r.div, "profit": odds - 1 if won else -1.0,
                             "clv": clv, "odds": odds})
    bets = pd.DataFrame(bets)

    def bet_stats(b):
        if b.empty:
            return {"bets": 0}
        return {"bets": int(len(b)),
                "roi": round(float(b.profit.mean()), 4),
                "profit_units": round(float(b.profit.sum()), 1),
                "avg_odds": round(float(b.odds.mean()), 2),
                "avg_clv": round(float(b.clv.mean()), 4) if b.clv.notna().any() else None,
                "beat_close_pct": round(float((b.clv > 0).mean()), 3) if b.clv.notna().any() else None}

    def block(mask):
        mk = mask & has
        mo = mask & has_ou
        return {
            "matches": int(mask.sum()),
            "logloss_1x2_model": round(float(log_loss(model_p[mk], res[mk])), 4) if mk.any() else None,
            "logloss_1x2_market": round(float(log_loss(mkt_p[mk], res[mk])), 4) if mk.any() else None,
            "brier_1x2_model": round(float(brier(model_p[mk], res[mk])), 4) if mk.any() else None,
            "brier_1x2_market": round(float(brier(mkt_p[mk], res[mk])), 4) if mk.any() else None,
            "logloss_ou25_model": round(float(log_loss(model_ou[mo], ou_res[mo])), 4) if mo.any() else None,
            "logloss_ou25_market": round(float(log_loss(mkt_ou[mo], ou_res[mo])), 4) if mo.any() else None,
        }

    out = {"version": config.VERSION,
           "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
           "from": str(bt.date.min().date()), "to": str(bt.date.max().date()),
           "min_edge": config.MIN_EDGE,
           "overall": {**block(np.ones(len(bt), bool)),
                       "value_bets": bet_stats(bets)},
           "leagues": {}}
    for div, name in config.LEAGUES.items():
        mask = (bt["div"] == div).to_numpy()
        if mask.any():
            out["leagues"][div] = {"name": name, **block(mask),
                                   "value_bets": bet_stats(bets[bets["div"] == div] if not bets.empty else bets)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=config.SEASONS[1])
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()

    matches = data.completed(data.load_results(refresh=not a.no_download))
    bt = run(matches, a.start)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    bt.to_csv(config.DATA_DIR / "backtest_matches.csv", index=False)
    summary = summarise(bt)
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DATA_DIR / "backtest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
