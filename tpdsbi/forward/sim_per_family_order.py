"""
sim_hydrogenated_carbon variant with per-family reaction order (mixed desorption mechanisms).
Families stay quasi-independent (cross-family recombination tested in
sensitivity_studies.coupled_recombination_bias). params (15).
"""
import numpy as np

KB = 8.617333e-5
T_GRID = np.linspace(250.0, 900.0, 200)
N_EA = 256
DTYPE = np.float32

LO = np.array([0.05, 0.05, 0.25, 0.05, 0.0, 0.25, 0.05, 0.0, 0.25, 0.05, 0.0,
               0.8, 0.8, 0.8, 0.0])
HI = np.array([0.90, 0.90, 2.20, 0.25, 14., 2.20, 0.25, 14., 2.20, 0.25, 14.,
               2.5, 2.5, 2.5, 0.15])
NP_ = 15
LOG_BETA_RANGE = (-1.0, 0.7)


def sample_prior(n, rng):
    p = LO + (HI - LO) * rng.random((n, NP_))
    w3 = rng.uniform(0.05, 0.90, n)
    s = p[:, 0] + p[:, 1] + w3
    p[:, 0] /= s; p[:, 1] /= s
    blocks = p[:, 2:11].reshape(n, 3, 3)
    order = np.argsort(blocks[:, :, 0], axis=1)
    p[:, 2:11] = np.take_along_axis(blocks, order[:, :, None], axis=1).reshape(n, 9)
    # per-family n rides with its (mu-sorted) family
    p[:, 11:14] = np.take_along_axis(p[:, 11:14], order, axis=1)
    return p


def simulate(params, log_beta, noise=0.0, rng=None):
    p = np.atleast_2d(params).astype(DTYPE)
    B = len(p)
    beta = (10.0 ** (np.full(B, log_beta) if np.isscalar(log_beta)
                     else np.asarray(log_beta))).astype(DTYPE)
    Ea = np.linspace(0.15, 2.6, N_EA, dtype=DTYPE)
    Tg = T_GRID.astype(DTYPE)
    dT = DTYPE(Tg[1] - Tg[0])
    boltz = np.exp(-Ea[:, None] / (KB * Tg[None, :]))
    I = np.cumsum(boltz, axis=1) * dT
    r = np.zeros((B, len(Tg)), dtype=DTYPE)
    w = np.column_stack([p[:, 0], p[:, 1], 1 - p[:, 0] - p[:, 1]])
    for c in range(3):
        mu, sig, lnu = p[:, 2 + 3*c], p[:, 3 + 3*c], p[:, 4 + 3*c]
        n_ord = p[:, 11 + c][:, None, None]
        nm1 = np.clip(n_ord - 1.0, 1e-3, None)
        nu_b = (10.0 ** lnu / beta)
        rho = np.exp(-0.5 * ((Ea[None, :] - mu[:, None]) / sig[:, None]) ** 2)
        rho /= rho.sum(1, keepdims=True)
        x = np.clip(nu_b[:, None, None] * I[None], 0, 80)
        theta_n = np.where(np.abs(n_ord - 1.0) < 0.02,
                           np.exp(-n_ord * x),
                           np.power(1.0 + nm1 * x, -n_ord / nm1))
        rate = nu_b[:, None, None] * theta_n * boltz[None]
        r += w[:, c, None] * (rho[:, :, None] * rate).sum(1)
    r /= r.max(1, keepdims=True) + 1e-12
    r = (r + p[:, 14:15]) / (1 + p[:, 14:15])
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        r = np.clip(r + noise * rng.standard_normal(r.shape).astype(DTYPE), 0, None)
    return r if np.asarray(params).ndim > 1 else r[0]
