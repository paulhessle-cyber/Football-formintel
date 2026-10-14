"""Project configuration."""
from pathlib import Path

VERSION = "0.4"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LOG_DIR = ROOT / "logs"
SITE_DATA_DIR = ROOT / "docs" / "data"

# football-data.co.uk division codes
LEAGUES = {
    "E2": "League One",
    "E3": "League Two",
    "EC": "National League",
}

# Leagues used for TRAINING only (not predicted or shown). Including the
# divisions above ours means teams relegated from the Championship (or
# further) arrive with up-to-date ratings instead of stale or no history,
# and keeps every team on one strength scale.
TRAIN_ONLY_LEAGUES = {
    "E0": "Premier League",
    "E1": "Championship",
}
TRAIN_LEAGUES = {**TRAIN_ONLY_LEAGUES, **LEAGUES}

# Seasons to download (football-data.co.uk "yyYY" codes). The first three are
# history; the last is the season in progress and is re-downloaded every run.
SEASONS = ["2324", "2425", "2526", "2627"]
CURRENT_SEASON = SEASONS[-1]

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Model settings
HALF_LIFE_DAYS = 240        # time decay: a match this old counts half as much
TRAIN_YEARS = 3             # ignore matches older than this
RIDGE = 0.5                 # pulls thin-data teams toward average
MAX_GOALS = 10              # score matrix size

# Value flagging
MIN_EDGE = 0.05             # model prob x odds - 1 must exceed this
