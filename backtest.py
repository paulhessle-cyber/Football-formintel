"""Walk-forward backtest against the closing line.

Every week the team model is refitted on matches played before that week and
predicts that week's matches, exactly as it would have live. The model is
then blended with the pre-match market using a weight fitted only on earlier
weeks (see fp/blend.py). We report:

  * log loss / Brier for model, pre-match market, blend and the closing line
    (Betfair Exchange close; Pinnacle / bookmaker average as fallbacks)
  * value bets: level stakes where the BLENDED probability gives an edge
    >= MIN_EDGE at the best pre-match price, with ROI and closing-line value

The final blend weights are saved to data/blend.json for predict.py.

Usage:  python backtest.py [--from 2425] [--no-download]
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from fp import blend, config, data, markets, model


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), 1e-12, 1)
    return float(-np.log(p[np.arange(len(p)), y]).mean())


def brier(p, y):
    p = np.asarray(p, float)
    t = np.zeros_like(p)
    t[np.arange(len(p)), y] = 1
    return float(((p - t) ** 2).sum(axis=1).mean())


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
                rows.append({**r.to_dict(), "week": week, "xg_h": lam, "xg_a": mu_,
                             "p_h": ph, "p_d": pd_, "p_a": pa,
                             "p_o25": po, "p_u25": pu})
        week = nxt
    return pd.DataFrame(rows)


def first_available(*arrays):
    """Row-wise: take the first array whose row is complete."""
    out = np.full(arrays[0].shape, np.nan)
    for a in arrays:
        need = np.isnan(out).any(axis=1) & np.all(np.isfinite(a), axis=1)
        out[need] = a[need]
    return out


def add_blend(bt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Walk-forward blend: each week uses a weight fitted on earlier weeks."""
    res = np.select([bt.hg > bt.ag, bt.hg == bt.ag], [0, 1], 2)
    ou = (bt.hg + bt.ag < 2.5).astype(int).to_numpy()        # 0 over, 1 under
    m1 = bt[["p_h", "p_d", "p_a"]].to_numpy()
    k1 = blend.devig_rows(bt[["avg_h", "avg_d", "avg_a"]])
    m2 = bt[["p_o25", "p_u25"]].to_numpy()
    k2 = blend.devig_rows(bt[["avg_o25", "avg_u25"]])

    b1, b2 = m1.copy(), m2.copy()
    w1s, w2s = np.zeros(len(bt)), np.zeros(len(bt))
    weeks = bt["week"].to_numpy()
    for wk in np.unique(weeks):
        past, now = weeks < wk, weeks == wk
        w1 = blend.fit_weight(m1[past], k1[past], res[past])
        w2 = blend.fit_weight(m2[past], k2[past], ou[past])
        b1[now] = blend.pool(m1[now], k1[now], w1)
        b2[now] = blend.pool(m2[now], k2[now], w2)
        w1s[now], w2s[now] = w1, w2
    bt = bt.assign(b_h=b1[:, 0], b_d=b1[:, 1], b_a=b1[:, 2],
                   b_o25=b2[:, 0], b_u25=b2[:, 1], w_1x2=w1s, w_ou25=w2s)
    final = {"w_1x2": blend.fit_weight(m1, k1, res),
             "w_ou25": blend.fit_weight(m2, k2, ou),
             "fitted_on": int(len(bt)),
             "generated": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    return bt, final


def summarise(bt: pd.DataFrame, weights: dict) -> dict:
    res = np.select([bt.hg > bt.ag, bt.hg == bt.ag], [0, 1], 2)
    ou_res = (bt.hg + bt.ag < 2.5).astype(int).to_numpy()

    model1 = bt[["p_h", "p_d", "p_a"]].to_numpy()
    blend1 = bt[["b_h", "b_d", "b_a"]].to_numpy()
    open1 = blend.devig_rows(bt[["avg_h", "avg_d", "avg_a"]])
    close1 = first_available(blend.devig_rows(bt[["bfec_h", "bfec_d", "bfec_a"]]),
                             blend.devig_rows(bt[["pc_h", "pc_d", "pc_a"]]),
                             blend.devig_rows(bt[["avgc_h", "avgc_d", "avgc_a"]]))
    model2 = bt[["p_o25", "p_u25"]].to_numpy()
    blend2 = bt[["b_o25", "b_u25"]].to_numpy()
    open2 = blend.devig_rows(bt[["avg_o25", "avg_u25"]])
    close2 = first_available(blend.devig_rows(bt[["bfec_o25", "bfec_u25"]]),
                             blend.devig_rows(bt[["pc_o25", "pc_u25"]]),
                             blend.devig_rows(bt[["avgc_o25", "avgc_u25"]]))
    has1 = np.all(np.isfinite(open1), 1) & np.all(np.isfinite(close1), 1)
    has2 = np.all(np.isfinite(open2), 1) & np.all(np.isfinite(close2), 1)

    # value bets from the BLENDED probabilities at the best pre-match price
    best = np.column_stack([
        bt[c1].where(bt[c1].notna(), bt[c2]).to_numpy(float)
        for c1, c2 in (("max_h", "avg_h"), ("max_d", "avg_d"), ("max_a", "avg_a"),
                       ("max_o25", "avg_o25"), ("max_u25", "avg_u25"))])
    probs = np.column_stack([blend1, blend2])
    closes = np.column_stack([close1, close2])
    won = np.column_stack([res[:, None] == np.arange(3), ou_res[:, None] == np.arange(2)])
    edge = probs * best - 1
    sel = np.isfinite(edge) & (edge >= config.MIN_EDGE)
    i, k = np.where(sel)
    bets = pd.DataFrame({
        "div": bt["div"].to_numpy()[i],
        "market": np.where(k < 3, "1x2", "ou25"),
        "odds": best[i, k],
        "profit": np.where(won[i, k], best[i, k] - 1, -1.0),
        "clv": best[i, k] * closes[i, k] - 1,
    })

    def r4(x):
        return None if x is None or not np.isfinite(x) else round(float(x), 4)

    def bet_stats(b):
        if b.empty:
            return {"bets": 0}
        return {"bets": int(len(b)),
                "roi": r4(b.profit.mean()),
                "profit_units": round(float(b.profit.sum()), 1),
                "avg_odds": round(float(b.odds.mean()), 2),
                "avg_clv": r4(b.clv.mean()),
                "beat_close_pct": round(float((b.clv > 0).mean()), 3)}

    def block(mask):
        a, b = mask & has1, mask & has2
        out = {"matches": int(mask.sum())}
        if a.any():
            y = res[a]
            out.update(logloss_1x2_model=r4(log_loss(model1[a], y)),
                       logloss_1x2_open=r4(log_loss(open1[a], y)),
                       logloss_1x2_blend=r4(log_loss(blend1[a], y)),
                       logloss_1x2_close=r4(log_loss(close1[a], y)),
                       brier_1x2_blend=r4(brier(blend1[a], y)),
                       brier_1x2_close=r4(brier(close1[a], y)))
        if b.any():
            y = ou_res[b]
            out.update(logloss_ou25_model=r4(log_loss(model2[b], y)),
                       logloss_ou25_open=r4(log_loss(open2[b], y)),
                       logloss_ou25_blend=r4(log_loss(blend2[b], y)),
                       logloss_ou25_close=r4(log_loss(close2[b], y)))
        return out

    out = {"version": config.VERSION,
           "generated": datetime.now(timezone.utc).isoformat(timespec="minutes"),
           "from": str(bt.date.min().date()), "to": str(bt.date.max().date()),
           "min_edge": config.MIN_EDGE,
           "weights": {"w_1x2": weights["w_1x2"], "w_ou25": weights["w_ou25"]},
           "overall": {**block(np.ones(len(bt), bool)), "value_bets": bet_stats(bets)},
           "leagues": {}}
    for div, name in config.LEAGUES.items():
        mask = (bt["div"] == div).to_numpy()
        if mask.any():
            out["leagues"][div] = {"name": name, **block(mask),
                                   "value_bets": bet_stats(bets[bets["div"] == div])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default=config.SEASONS[1])
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()

    matches = data.completed(data.load_results(refresh=not a.no_download))
    bt = run(matches, a.start)
    bt, weights = add_blend(bt)
    blend.save(weights)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    bt.to_csv(config.DATA_DIR / "backtest_matches.csv", index=False)
    summary = summarise(bt, weights)
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (config.SITE_DATA_DIR / "backtest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
