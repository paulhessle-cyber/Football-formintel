# Lower League Predictor — v0.8

Match predictions for **League One, League Two and the National League**, built in layers so each one has to prove it improves accuracy before it's trusted.

| Layer | Status |
|---|---|
| 1. Team model (time-weighted Dixon-Coles; goals + shots on target) | ✅ v0.7 |
| 1b. Market blending — model must earn its weight | ✅ v0.4 |
| 2. News tags from match previews (scheduled, twice a week) | ✅ v0.6 — scored by the news scorecard |
| 3. Lineup / availability adjustments | planned |
| 4. Betfair Exchange prices from the FormIntel odds proxy (`server/football.js`) | ✅ v0.8 |

Markets: 1X2, Over/Under 2.5, Asian handicap (fair price at the market line).

## How it works

- **Data:** football-data.co.uk results and odds CSVs, last 3 seasons + current. Completed seasons are cached in `data/raw/`. The Premier League and Championship are used for **training only**, so teams relegated into League One arrive with current ratings. Only League One, League Two and the National League are predicted and scored.
- **Model:** one joint model across all five divisions, so promoted and relegated teams keep their rating and strengths are on one scale. Each team has an attack and defence rating; home advantage is per league. Old matches decay (half-life 240 days) and a ridge penalty stops thin-data teams going extreme. Teams with no history (promoted from National League North/South) are rated at the 25th percentile of their new league.
- **Fair prices:** bookmaker odds are converted to probabilities with the *power method*, which removes the favourite-longshot bias (plain scaling overrates outsiders). The model is then blended in with a weight learned in the backtest (currently ~0%), and news tags shift the result.
- **Prices you can get:** "Best UK" is the best price among the UK bookmakers in the data file (Bet365, Betfred, BetVictor, bwin, Paddy Power, Sky Bet…), with the bookmaker named. The file's "market maximum" is ignored for value flags because it is usually not at a UK bookmaker. Value = edge ≥ 5% at a UK price; betting at the best UK price with no edge loses about 5.5% on average.
- **Benchmark:** the **Betfair Exchange closing price** (Pinnacle stopped supplying data in 2025). The backtest refits weekly and only uses matches before each week. It reports log loss vs the closing line, plus value-bet ROI and **closing-line value (CLV)**. CLV is the number to trust; ROI over a few dozen bets is mostly luck.
- **News scorecard:** every match with news tags is scored on whether the bookmakers' price later moved the way the tags predicted (`logs/news_log.csv`). This judges the news layer on every tagged match, not just the rare value bets.
- **Logging:** every prediction is appended to `logs/predictions_log.csv` before kick-off, with and without news adjustments, so each layer can be scored honestly later.

## Betfair prices (odds proxy on the droplet)

`server/football.js` is an add-on for the FormIntel odds proxy (`odds.formintel.co.uk`). It shares the proxy's Betfair login (a delayed key is fine) and records Match Odds and Over/Under 2.5 prices for every League One, League Two and National League match: every 30 min from 3 days out, every 10 min in the last 6 hours, every 2 min in the last hour. `GET /football?status=1` shows what it's doing.

Install once, on the droplet, from the proxy's folder:

```
cd /root/formintel-proxy
curl -fsSLO https://raw.githubusercontent.com/paulhessle-cyber/Football-formintel/main/server/football.js
curl -fsSLO https://raw.githubusercontent.com/paulhessle-cyber/Football-formintel/main/server/install-football.js
node install-football.js
```

then restart the proxy as the installer says. Each predictions run fetches `/football?days=21` and keeps a summary per match in `data/betfair/` (first, 24h, 6h, 1h, 10 min before, and the closing price). The site shows the current Betfair price and how far it has moved; paper bets and the news scorecard use the recorded Betfair closing price.

## Setup (one time)

1. Create a new GitHub repo and upload this folder's contents.
2. **Settings → Pages →** Source: *Deploy from a branch*, Branch: `main`, Folder: `/docs`.
3. **Settings → Actions → General →** Workflow permissions: *Read and write*.
4. **Actions tab →** run *Weekly backtest*, then *Daily predictions* (both have a *Run workflow* button).

After that it runs itself: predictions several times a day, backtest every Monday.

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

- **v0.8** — Betfair Exchange prices recorded by the odds proxy on the droplet: current price and movement on each fixture, true Betfair closing price for paper bets and the news scorecard.
- **v0.7** — Fairer odds conversion (power method) removes fake value on long shots. Value flags and paper bets now use the best price at a named UK bookmaker, with a sanity filter for mistimed prices. Team ratings use shots on target as well as goals (1X2 log loss 1.045 → 1.040). Asian handicap value is based on the market's own handicap prices. News scorecard added. Four schedule slots a day to cope with GitHub's late starts. Blend weights are now saved by the weekly backtest. Earlier paper bets are kept but reported separately.
- **v0.6** — Daily predictions also run whenever `data/news_events.csv` is updated, so new news tags take effect within minutes. Twice-weekly scheduled news tagging from match previews.
- **v0.5** — Paper bets tab: every value flag is logged at 1 unit the first time it appears (best and average price), settled automatically from results, with closing-line value. Split by news-tagged vs not, and by market. Log: `logs/paper_bets.csv`.
- **v0.4** — Model is blended with the bookmakers' pre-match price using a weight learned from the backtest (currently ~0%: the goals-only model adds nothing the market lacks). News tags shift the final odds at full strength. Benchmark switched to Betfair Exchange closing prices (Pinnacle stopped supplying football-data in 2025). Track record shows Model / Market / Blend / Close side by side.
- **v0.3** — Workflows no longer fail when two runs save results at the same time (newest results win).
- **v0.2** — Premier League and Championship added as training-only data. Fixes recently relegated sides (e.g. Oxford) being rated from stale League One form.
- **v0.1** — Baseline Dixon-Coles model, walk-forward backtest, daily predictions, manual news layer, website.

## Forcing a refresh

GitHub often runs scheduled jobs hours late. Editing `data/refresh.txt` (any change, e.g. the current time) and pushing starts Daily predictions immediately.
