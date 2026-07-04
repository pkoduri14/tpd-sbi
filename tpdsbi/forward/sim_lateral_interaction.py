"""
Rival single-site model for the lateral-interaction comparison (sensitivity_studies.py):
coverage-dependent Ea(theta) = E0 - alpha*theta (Bragg-Williams mean field), broadening the
peak without a site distribution. First-order; sequential T-grid integration (SUB sub-steps
per cell), vectorized over the batch, no (B,E,T) tensor. params (4).
"""
import numpy as np

KB = 8.617333e-5
T_GRID = np.linspace(250.0, 900.0, 200)
DTYPE = np.float32

LO = np.array([0.50, 0.00, 0.0, 0.00])
HI = np.array([2.60, 1.50, 14., 0.15])
NP_ = 4
LOG_BETA_RANGE = (-1.0, 0.7)
SUB = 128  # alpha-feedback stiffness; verified vs SUB=512/f64


def sample_prior(n, rng):
    return LO + (HI - LO) * rng.random((n, NP_))


def simulate(params, log_beta, noise=0.0, rng=None):
    p = np.atleast_2d(params).astype(DTYPE)
    B = len(p)
    beta = (10.0 ** (np.full(B, log_beta) if np.isscalar(log_beta)
                     else np.asarray(log_beta))).astype(DTYPE)
    E0, al, lnu = p[:, 0], p[:, 1], p[:, 2]
    nu_b = 10.0 ** lnu / beta
    Tg = T_GRID.astype(DTYPE)
    dT = DTYPE(Tg[1] - Tg[0]) / SUB
    theta = np.ones(B, dtype=DTYPE)
    r = np.zeros((B, len(Tg)), dtype=DTYPE)
    for t in range(len(Tg)):
        rate_out = np.zeros(B, dtype=DTYPE)
        for s in range(SUB):
            T = Tg[t] + (s - (SUB - 1) / 2) * dT
            Ea = np.clip(E0 - al * theta, 0.05, None)
            k = nu_b * np.exp(-Ea / (KB * T))
            d = np.clip(k * theta * dT, 0, theta)   # bounded Euler; small steps handle stiffness
            theta = theta - d
            rate_out += d
        r[:, t] = rate_out
    r /= r.max(1, keepdims=True) + 1e-12
    r = (r + p[:, 3:4]) / (1 + p[:, 3:4])
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        r = np.clip(r + noise * rng.standard_normal(r.shape).astype(DTYPE), 0, None)
    return r if np.asarray(params).ndim > 1 else r[0]
