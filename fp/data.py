"""Download and normalise football-data.co.uk CSVs."""
import io
import time

import numpy as np
import pandas as pd
import requests

from . import config

# Columns we keep, mapped from football-data names to our own.
ODDS_COLS = {
    # pre-match (opening-ish) prices
    "AvgH": "avg_h", "AvgD": "avg_d", "AvgA": "avg_a",
    "MaxH": "max_h", "MaxD": "max_d", "MaxA": "max_a",
    "Avg>2.5": "avg_o25", "Avg<2.5": "avg_u25",
    "Max>2.5": "max_o25", "Max<2.5": "max_u25",
    "AHh": "ah_line", "AvgAHH": "avg_ahh", "AvgAHA": "avg_aha",
    "MaxAHH": "max_ahh", "MaxAHA": "max_aha",
    # closing prices (the benchmark)
    "PSCH": "pc_h", "PSCD": "pc_d", "PSCA": "pc_a",
    "AvgCH": "avgc_h", "AvgCD": "avgc_d", "AvgCA": "avgc_a",
    "MaxCH": "maxc_h", "MaxCD": "maxc_d", "MaxCA": "maxc_a",
    "PC>2.5": "pc_o25", "PC<2.5": "pc_u25",
    "AvgC>2.5": "avgc_o25", "AvgC<2.5": "avgc_u25",
    "MaxC>2.5": "maxc_o25", "MaxC<2.5": "maxc_u25",
    "AHCh": "ahc_line", "PCAHH": "pc_ahh", "PCAHA": "pc_aha",
    "AvgCAHH": "avgc_ahh", "AvgCAHA": "avgc_aha",
    "MaxCAHH": "maxc_ahh", "MaxCAHA": "maxc_aha",
    # Betfair Exchange (sharp, still published after Pinnacle stopped in 2025)
    "BFEH": "bfe_h", "BFED": "bfe_d", "BFEA": "bfe_a",
    "BFECH": "bfec_h", "BFECD": "bfec_d", "BFECA": "bfec_a",
    "BFE>2.5": "bfe_o25", "BFE<2.5": "bfe_u25",
    "BFEC>2.5": "bfec_o25", "BFEC<2.5": "bfec_u25",
    # match xG (football-data added this in 2026/27)
    "HxG": "hxg", "AxG": "axg",
    # shots / shots on target (not published for the National League)
    "HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast",
}

# Bookmakers in the football-data files that take UK customers. Their prices
# are the ones you could actually bet at; "Max" in the files is the highest
# price from a wider panel and is usually NOT at one of these.
UK_BOOKS = {
    "B365": "Bet365", "BFD": "Betfred", "BMGM": "BetMGM", "BV": "BetVictor",
    "BW": "bwin", "CL": "Coral", "LB": "Ladbrokes", "PP": "Paddy Power",
    "SKB": "Sky Bet", "WH": "William Hill", "BF": "Betfair",
}
# selection -> column suffix in the raw files
BOOK_SUFFIX = {"h": "H", "d": "D", "a": "A", "o25": ">2.5", "u25": "<2.5",
               "ahh": "AHH", "aha": "AHA"}
SELECTIONS = list(BOOK_SUFFIX)


def _uk_best(raw: pd.DataFrame) -> dict:
    """Best pre-match price among UK bookmakers for each selection, and who."""
    cols = {}
    for sel, suf in BOOK_SUFFIX.items():
        have = [(code, name) for code, name in UK_BOOKS.items() if code + suf in raw.columns]
        if not have:
            cols[f"uk_{sel}"] = np.nan
            cols[f"ukb_{sel}"] = ""
            continue
        prices = np.column_stack([pd.to_numeric(raw[c + suf], errors="coerce").to_numpy(float)
                                  for c, _ in have])
        prices[~(prices > 1)] = np.nan
        # Sanity check: a bookmaker price above the file's own "market maximum"
        # was collected at a different time (seen with William Hill in 2024/25)
        # and isn't a real opportunity, so ignore it.
        if "Max" + suf in raw.columns:
            cap = pd.to_numeric(raw["Max" + suf], errors="coerce").to_numpy(float) * 1.011
            prices[prices > np.where(np.isfinite(cap), cap, np.inf)[:, None]] = np.nan
        filled = np.where(np.isfinite(prices), prices, -1.0)
        idx = filled.argmax(axis=1)
        best = filled.max(axis=1)
        names = np.array([n for _, n in have], dtype=object)[idx]
        cols[f"uk_{sel}"] = np.where(best > 1, best, np.nan)
        cols[f"ukb_{sel}"] = np.where(best > 1, names, "")
    return cols


def _get(url: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30,
                             headers={"User-Agent": "football-predictor"})
            r.raise_for_status()
            return r.content.decode("utf-8-sig", errors="replace")
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return ""


def normalise(raw: pd.DataFrame, div: str | None = None) -> pd.DataFrame:
    """Turn a football-data frame into our standard shape."""
    raw = raw.dropna(how="all")
    raw = raw.rename(columns=lambda c: str(c).strip())
    if "HomeTeam" not in raw.columns and "Home" in raw.columns:
        raw = raw.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam",
                                  "HG": "FTHG", "AG": "FTAG"})
    out = pd.DataFrame()
    out["div"] = raw["Div"] if "Div" in raw.columns else div
    out["date"] = pd.to_datetime(raw["Date"], dayfirst=True, format="mixed",
                                 errors="coerce")
    out["time"] = raw["Time"] if "Time" in raw.columns else ""
    out["home"] = raw["HomeTeam"].astype(str).str.strip()
    out["away"] = raw["AwayTeam"].astype(str).str.strip()
    for src, dst in (("FTHG", "hg"), ("FTAG", "ag")):
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else np.nan
    extra = {dst: (pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else np.nan)
             for src, dst in ODDS_COLS.items()}
    extra.update(_uk_best(raw))
    out = pd.concat([out, pd.DataFrame(extra, index=raw.index)], axis=1)
    out = out.dropna(subset=["date"])
    out = out[(out["home"] != "nan") & (out["away"] != "nan")]
    return out.reset_index(drop=True)


def download_results(refresh_current: bool = True) -> pd.DataFrame:
    """Fetch every league/season, caching completed seasons on disk."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season in config.SEASONS:
        for div in config.TRAIN_LEAGUES:
            path = config.RAW_DIR / f"{season}_{div}.csv"
            fresh = season == config.CURRENT_SEASON and refresh_current
            if fresh or not path.exists():
                url = config.BASE_URL.format(season=season, div=div)
                try:
                    path.write_text(_get(url), encoding="utf-8")
                    print(f"downloaded {season} {div}")
                except requests.RequestException as e:
                    print(f"WARNING: could not download {url}: {e}")
                    if not path.exists():
                        continue
            df = normalise(pd.read_csv(path, encoding="utf-8",
                                       on_bad_lines="skip"), div)
            df["season"] = season
            frames.append(df)
    if not frames:
        raise RuntimeError("No results data available")
    return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)


def load_results(refresh: bool = True) -> pd.DataFrame:
    df = download_results(refresh_current=refresh)
    return df


def download_fixtures() -> pd.DataFrame:
    """Upcoming fixtures (with current odds) for our leagues."""
    text = _get(config.FIXTURES_URL)
    raw = pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    raw = raw.rename(columns=lambda c: str(c).strip())
    raw = raw[raw["Div"].isin(config.LEAGUES.keys())]
    return normalise(raw)


def completed(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["hg", "ag"]).reset_index(drop=True)


def best_price(r, sel: str):
    """(price, bookmaker) for a selection: best UK bookmaker, else market max."""
    uk = r.get(f"uk_{sel}", np.nan)
    if pd.notna(uk) and uk > 1:
        return float(uk), str(r.get(f"ukb_{sel}", ""))
    for col in (f"max_{sel}", f"avg_{sel}"):
        v = r.get(col, np.nan)
        if pd.notna(v) and v > 1:
            return float(v), "market max" if col.startswith("max") else "market avg"
    return np.nan, ""
