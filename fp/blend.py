"""Blend model probabilities with the betting market.

The market (de-vigged average bookmaker price) already contains most of
what a results-based model knows. Instead of replacing it, the model gets
a weight w learned from the backtest:

    p_final  ∝  p_market^(1-w) * p_model^w        (logarithmic pooling)

w = 0 means "trust the market", w = 1 means "trust the model". w is refitted
every week from past matches, so any new layer (xG, news, lineups) only gains
influence once it has been shown to improve predictions.
"""
import json

import numpy as np
from scipy.optimize import minimize_scalar

from . import config

BLEND_FILE = config.DATA_DIR / "blend.json"
DEFAULT = {"w_1x2": 0.0, "w_ou25": 0.0}


def devig_rows(odds: np.ndarray) -> np.ndarray:
    """Rows of decimal odds -> de-vigged probabilities (NaN if incomplete)."""
    odds = np.asarray(odds, float)
    out = np.full(odds.shape, np.nan)
    ok = np.all(np.isfinite(odds) & (odds > 1), axis=1)
    inv = 1 / odds[ok]
    out[ok] = inv / inv.sum(axis=1, keepdims=True)
    return out


def pool(model: np.ndarray, market: np.ndarray, w: float) -> np.ndarray:
    """Logarithmic pool. Rows without a market price fall back to the model."""
    model = np.clip(np.asarray(model, float), 1e-9, 1)
    market = np.asarray(market, float)
    out = model.copy()
    ok = np.all(np.isfinite(market), axis=1)
    q = np.clip(market[ok], 1e-9, 1) ** (1 - w) * model[ok] ** w
    out[ok] = q / q.sum(axis=1, keepdims=True)
    return out


def fit_weight(model: np.ndarray, market: np.ndarray, outcome: np.ndarray) -> float:
    """Weight minimising log loss on past matches (0 if too little data)."""
    ok = np.all(np.isfinite(market), axis=1) & np.all(np.isfinite(model), axis=1)
    if ok.sum() < 300:
        return 0.0
    m, k, y = model[ok], market[ok], outcome[ok]
    idx = np.arange(len(y))

    def ll(w):
        return -np.log(pool(m, k, w)[idx, y]).mean()

    w = minimize_scalar(ll, bounds=(0, 1), method="bounded").x
    return float(round(w, 3)) if w > 0.005 else 0.0


def load() -> dict:
    try:
        return {**DEFAULT, **json.loads(BLEND_FILE.read_text())}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT)


def save(weights: dict):
    BLEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    BLEND_FILE.write_text(json.dumps(weights, indent=2))


def fit_matrix(m: np.ndarray, p1x2, pou=None, iters: int = 30) -> np.ndarray:
    """Rescale a score matrix so its 1X2 (and O/U 2.5) match target probs.

    Iterative proportional fitting: keeps the model's scoreline shape (for
    Asian handicaps etc.) while agreeing with the blended market totals.
    """
    g = np.arange(m.shape[0])
    margin = g[:, None] - g[None, :]
    total = g[:, None] + g[None, :]
    res_masks = [margin > 0, margin == 0, margin < 0]
    ou_masks = [total > 2.5, total < 2.5]
    m = m.copy()
    for _ in range(iters):
        for mask, target in zip(res_masks, p1x2):
            s = m[mask].sum()
            if s > 0:
                m[mask] *= target / s
        if pou is None:
            break
        for mask, target in zip(ou_masks, pou):
            s = m[mask].sum()
            if s > 0:
                m[mask] *= target / s
    return m / m.sum()
