"""Turn a score matrix into market probabilities and fair prices."""
import numpy as np


def one_x_two(m: np.ndarray) -> tuple[float, float, float]:
    home = float(np.tril(m, -1).sum())
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())
    return home, draw, away


def over_under(m: np.ndarray, line: float = 2.5) -> tuple[float, float]:
    g = np.arange(m.shape[0])
    total = g[:, None] + g[None, :]
    over = float(m[total > line].sum())
    under = float(m[total < line].sum())
    return over, under


def _ah_parts(line: float) -> list[float]:
    """Split quarter lines (e.g. -0.75) into two half-stake lines."""
    q = round(line * 4)
    if q % 2:            # quarter line
        return [(q - 1) / 4, (q + 1) / 4]
    return [line]


def ah_outcomes(m: np.ndarray, line: float, side: str = "home"):
    """[(stake_fraction, p_win, p_push, p_lose)] for an Asian handicap bet.

    `line` is the HOME handicap (football-data's AHh convention): -0.5 means
    home gives half a goal. For side='away' the away team gets -line.
    """
    g = np.arange(m.shape[0])
    margin = g[:, None] - g[None, :]          # home goals - away goals
    if side == "away":
        margin = -margin
        line = -line
    parts = _ah_parts(line)
    out = []
    for part in parts:
        adj = margin + part
        out.append((1 / len(parts), float(m[adj > 0].sum()),
                    float(m[adj == 0].sum()), float(m[adj < 0].sum())))
    return out


def ah_ev(m: np.ndarray, line: float, odds: float, side: str = "home") -> float:
    """Expected profit per 1 unit staked."""
    return sum(f * (pw * (odds - 1) - pl)
               for f, pw, _, pl in ah_outcomes(m, line, side))


def ah_fair_odds(m: np.ndarray, line: float, side: str = "home") -> float:
    parts = ah_outcomes(m, line, side)
    win = sum(f * pw for f, pw, _, _ in parts)
    lose = sum(f * pl for f, _, _, pl in parts)
    return float("inf") if win <= 0 else 1 + lose / win


def ah_fair_line(m: np.ndarray) -> tuple[float, float, float]:
    """Home handicap whose fair home price is closest to evens."""
    best = None
    for line in np.arange(-3, 3.01, 0.25):
        o = ah_fair_odds(m, float(line))
        if best is None or abs(o - 2) < abs(best[1] - 2):
            best = (float(line), o, ah_fair_odds(m, float(line), "away"))
    return best


def devig(*odds: float) -> list[float] | None:
    """Bookmaker odds -> probabilities, margin removed proportionally."""
    if any(o is None or not np.isfinite(o) or o <= 1 for o in odds):
        return None
    inv = np.array([1 / o for o in odds])
    return list(inv / inv.sum())


def fair(p: float) -> float | None:
    return round(1 / p, 2) if p > 0 else None


def edge(p: float, odds: float) -> float | None:
    if odds is None or not np.isfinite(odds) or odds <= 1:
        return None
    return p * odds - 1
