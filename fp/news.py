"""Manual news-event adjustments (step 3 of the roadmap, usable now).

Each row in data/news_events.csv nudges a team's expected goals for matches
between `date_added` and `expires` (blank expiry = 14 days):

  attack_pct   -10  -> team scores 10% fewer expected goals
  defence_pct   -8  -> team's defence is 8% worse (opponent scores 8% more)
  confidence   0-1  -> scales the effect (confirmed = 1, rumour = 0.3)

Predictions are logged both with and without these adjustments, so the
news layer can be judged on whether it actually improves accuracy.
"""
import pandas as pd

from . import config

NEWS_FILE = config.DATA_DIR / "news_events.csv"
COLUMNS = ["date_added", "team", "player", "kind", "attack_pct",
           "defence_pct", "confidence", "expires", "note"]


def load() -> pd.DataFrame:
    if not NEWS_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(NEWS_FILE, comment="#", skip_blank_lines=True)
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["expires"] = pd.to_datetime(df["expires"], errors="coerce")
    df["expires"] = df["expires"].fillna(df["date_added"] + pd.Timedelta(days=14))
    for c in ("attack_pct", "defence_pct"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(1.0).clip(0, 1)
    df["team"] = df["team"].astype(str).str.strip()
    return df.dropna(subset=["date_added"])


def multipliers(events: pd.DataFrame, home: str, away: str,
                match_date: pd.Timestamp) -> tuple[float, float, list[str]]:
    """xG multipliers for (home, away) and the notes that were applied."""
    hm, am, notes = 1.0, 1.0, []
    if events.empty:
        return hm, am, notes
    live = events[(events["date_added"] <= match_date) &
                  (events["expires"] >= match_date.normalize())]
    for _, e in live.iterrows():
        if e["team"] not in (home, away):
            continue
        c = e["confidence"]
        own = 1 + e["attack_pct"] / 100 * c
        opp = 1 - e["defence_pct"] / 100 * c
        if e["team"] == home:
            hm *= own
            am *= opp
        else:
            am *= own
            hm *= opp
        who = f" ({e['player']})" if isinstance(e["player"], str) and e["player"] else ""
        notes.append(f"{e['team']}{who}: {e['kind']}"
                     + (f" - {e['note']}" if isinstance(e["note"], str) and e["note"] else ""))
    return hm, am, notes
