# Lower League Predictor — v0.2

Match predictions for **League One, League Two and the National League**, built in layers so each one has to prove it improves accuracy before it's trusted.

| Layer | Status |
|---|---|
| 1. Baseline team model (time-weighted Dixon-Coles Poisson) | ✅ v0.1 |
| 2. Lineup / availability adjustments | planned |
| 3. News events (manual tagging) | ✅ scaffolded — `data/news_events.csv` |
| 4. Automated news tagging | later, once manual tags show value |

Markets: 1X2, Over/Under 2.5, Asian handicap (fair line + edge vs market line).

## How it works

- **Data:** football-data.co.uk results and odds CSVs, last 3 seasons + current. Completed seasons are cached in `data/raw/`. The Premier League and Championship are used for **training only**, so teams relegated into League One arrive with current ratings. Only League One, League Two and the National League are predicted and scored.
- **Model:** one joint model across all five divisions, so promoted and relegated teams keep their rating and strengths are on one scale. Each team has an attack and defence rating; home advantage is per league. Old matches decay (half-life 240 days) and a ridge penalty stops thin-data teams going extreme. Teams with no history (promoted from National League North/South) are rated at the 25th percentile of their new league.
- **Benchmark:** the de-vigged **Pinnacle closing price**. The backtest refits weekly and only uses matches before each week. It reports log loss and Brier vs the closing line, plus value-bet ROI and **closing-line value (CLV)**. CLV is the number to trust; ROI over a few hundred bets is mostly noise.
- **Logging:** every prediction is appended to `logs/predictions_log.csv` before kick-off, with and without news adjustments, so each layer can be scored honestly later.

## Setup (one time)

1. Create a new GitHub repo and upload this folder's contents.
2. **Settings → Pages →** Source: *Deploy from a branch*, Branch: `main`, Folder: `/docs`.
3. **Settings → Actions → General →** Workflow permissions: *Read and write*.
4. **Actions tab →** run *Weekly backtest*, then *Daily predictions* (both have a *Run workflow* button).

After that it runs itself: predictions twice a day, backtest every Monday.

## Adding news

Edit `data/news_events.csv` (works from the GitHub app/website):

```
date_added,team,player,kind,attack_pct,defence_pct,confidence,expires,note
2026-10-14,Example Town,J Smith,injury - top scorer out,-12,0,1.0,2026-11-01,club confirmed
```

- `attack_pct` −12 → team's expected goals down 12%
- `defence_pct` −5 → defence 5% worse (opponent's expected goals up 5%)
- `confidence` 1.0 confirmed, ~0.3 rumour; scales the effect
- blank `expires` → 14 days
- team names must match football-data.co.uk spelling (see the Ratings tab)

## Running locally

```
pip install -r requirements.txt
python backtest.py        # walk-forward test → docs/data/backtest.json
python predict.py         # upcoming fixtures → docs/data/predictions.json
```

Tunables are in `fp/config.py` (seasons, half-life, ridge, edge threshold). Bump `VERSION` there on every change.

## Files

```
fp/config.py      settings + version
fp/data.py        download & normalise football-data CSVs
fp/model.py       Dixon-Coles team model
fp/markets.py     1X2, O/U, Asian handicap maths
fp/news.py        news-event adjustments
predict.py        daily run
backtest.py       walk-forward evaluation
docs/index.html   website (GitHub Pages)
```

## Changelog

- **v0.2** — Premier League and Championship added as training-only data. Fixes recently relegated sides (e.g. Oxford) being rated from stale League One form.
- **v0.1** — Baseline Dixon-Coles model, walk-forward backtest, daily predictions, manual news layer, website.
