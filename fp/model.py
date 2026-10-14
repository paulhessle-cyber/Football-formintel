"""Baseline team-strength model: time-weighted Dixon-Coles Poisson.

One joint model across all three leagues, so promoted/relegated teams carry
their rating with them and strengths are comparable between divisions.

  log(home goals rate) = mu + home_adv[league] + attack[home] - defence[away]
  log(away goals rate) = mu +                    attack[away] - defence[home]

plus the Dixon-Coles low-score correction (rho). Older matches are
down-weighted exponentially; a ridge penalty keeps thin-data teams sensible.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from . import config


@dataclass
class TeamModel:
    teams: list[str]
    leagues: list[str]
    mu: float
    home_adv: dict[str, float]
    attack: dict[str, float]
    defence: dict[str, float]
    rho: float
    fitted_to: pd.Timestamp
    team_league: dict[str, str] = field(default_factory=dict)

    # ---------- prediction ----------
    def _rating(self, team: str, league: str) -> tuple[float, float]:
        if team in self.attack:
            return self.attack[team], self.defence[team]
        # Unknown (newly promoted from below our data): assume a weak side
        # in this league — the 25th percentile of its current members.
        members = [t for t, lg in self.team_league.items() if lg == league]
        if not members:
            return 0.0, 0.0
        att = np.percentile([self.attack[t] for t in members], 25)
        dfn = np.percentile([self.defence[t] for t in members], 25)
        return float(att), float(dfn)

    def rates(self, home: str, away: str, league: str,
              home_mult: float = 1.0, away_mult: float = 1.0) -> tuple[float, float]:
        """Expected goals for each side. *_mult are news/lineup adjustments."""
        ah, dh = self._rating(home, league)
        aa, da = self._rating(away, league)
        hadv = self.home_adv.get(league, np.mean(list(self.home_adv.values())))
        lam = np.exp(self.mu + hadv + ah - da) * home_mult
        mu_ = np.exp(self.mu + aa - dh) * away_mult
        return float(lam), float(mu_)

    def score_matrix(self, lam: float, mu_: float) -> np.ndarray:
        g = np.arange(config.MAX_GOALS + 1)
        m = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu_))
        r = self.rho
        m[0, 0] *= 1 - lam * mu_ * r
        m[0, 1] *= 1 + lam * r
        m[1, 0] *= 1 + mu_ * r
        m[1, 1] *= 1 - r
        m = np.clip(m, 0, None)
        return m / m.sum()


def _weights(dates: pd.Series, as_of: pd.Timestamp) -> np.ndarray:
    age = (as_of - dates).dt.days.to_numpy(dtype=float)
    xi = np.log(2) / config.HALF_LIFE_DAYS
    return np.exp(-xi * np.clip(age, 0, None))


def fit(matches: pd.DataFrame, as_of: pd.Timestamp | None = None,
        warm: TeamModel | None = None) -> TeamModel:
    """Fit on completed matches strictly before `as_of`."""
    if as_of is None:
        as_of = matches["date"].max() + pd.Timedelta(days=1)
    df = matches[(matches["date"] < as_of) &
                 (matches["date"] >= as_of - pd.DateOffset(years=config.TRAIN_YEARS))]
    df = df.dropna(subset=["hg", "ag"])
    if len(df) < 50:
        raise ValueError("Not enough matches to fit")

    teams = sorted(set(df["home"]) | set(df["away"]))
    leagues = sorted(df["div"].unique())
    t_idx = {t: i for i, t in enumerate(teams)}
    l_idx = {l: i for i, l in enumerate(leagues)}
    n, L = len(teams), len(leagues)

    hi = df["home"].map(t_idx).to_numpy()
    ai = df["away"].map(t_idx).to_numpy()
    li = df["div"].map(l_idx).to_numpy()
    hg = df["hg"].to_numpy(float)
    ag = df["ag"].to_numpy(float)
    w = _weights(df["date"], as_of)

    # params: [mu, home_adv(L), attack(n), defence(n)]
    def unpack(p):
        return p[0], p[1:1 + L], p[1 + L:1 + L + n], p[1 + L + n:]

    def nll(p):
        mu, ha, att, dfn = unpack(p)
        lh = mu + ha[li] + att[hi] - dfn[ai]
        la = mu + att[ai] - dfn[hi]
        eh, ea = np.exp(lh), np.exp(la)
        f = -np.sum(w * (hg * lh - eh + ag * la - ea))
        f += config.RIDGE * (att @ att + dfn @ dfn)
        rh = w * (hg - eh)   # d(loglik)/d(log rate home)
        ra = w * (ag - ea)
        g = np.zeros_like(p)
        g[0] = -(rh.sum() + ra.sum())
        g[1:1 + L] = -np.bincount(li, rh, L)
        g_att = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n))
        g_def = np.bincount(ai, rh, n) + np.bincount(hi, ra, n)
        g[1 + L:1 + L + n] = g_att + 2 * config.RIDGE * att
        g[1 + L + n:] = g_def + 2 * config.RIDGE * dfn
        return f, g

    p0 = np.zeros(1 + L + 2 * n)
    p0[0] = np.log(max(1e-3, (hg.mean() + ag.mean()) / 2))
    p0[1:1 + L] = 0.2
    if warm is not None:
        p0[0] = warm.mu
        for l, j in l_idx.items():
            p0[1 + j] = warm.home_adv.get(l, 0.2)
        for t, j in t_idx.items():
            p0[1 + L + j] = warm.attack.get(t, 0.0)
            p0[1 + L + n + j] = warm.defence.get(t, 0.0)

    res = minimize(nll, p0, jac=True, method="L-BFGS-B")
    mu, ha, att, dfn = unpack(res.x)

    # Dixon-Coles rho, fitted given the rates
    lam = np.exp(mu + ha[li] + att[hi] - dfn[ai])
    mu_r = np.exp(mu + att[ai] - dfn[hi])
    low = (hg <= 1) & (ag <= 1)

    def rho_nll(r):
        tau = np.ones_like(lam)
        m00 = low & (hg == 0) & (ag == 0)
        m01 = low & (hg == 0) & (ag == 1)
        m10 = low & (hg == 1) & (ag == 0)
        m11 = low & (hg == 1) & (ag == 1)
        tau[m00] = 1 - lam[m00] * mu_r[m00] * r
        tau[m01] = 1 + lam[m01] * r
        tau[m10] = 1 + mu_r[m10] * r
        tau[m11] = 1 - r
        if np.any(tau <= 0):
            return 1e12
        return -np.sum(w * np.log(tau))

    rho = minimize_scalar(rho_nll, bounds=(-0.25, 0.25), method="bounded").x

    # each team's most recent league
    last = df.sort_values("date")
    team_league = {}
    for _, r in last[["home", "away", "div"]].iterrows():
        team_league[r["home"]] = r["div"]
        team_league[r["away"]] = r["div"]

    return TeamModel(
        teams=teams, leagues=leagues, mu=float(mu),
        home_adv={l: float(ha[j]) for l, j in l_idx.items()},
        attack={t: float(att[j]) for t, j in t_idx.items()},
        defence={t: float(dfn[j]) for t, j in t_idx.items()},
        rho=float(rho), fitted_to=as_of, team_league=team_league,
    )
