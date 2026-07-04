"""
Forward TPD model for high-temperature carbon H2 desorption (Nechaev 2020 Fig. 3:
graphitized nanodiamonds, broad peak 1246-1298 K, beta = 0.17 K/s). Two site families
with per-family prefactor, free reaction order, baseline nuisance. params (9).
"""
import numpy as np

KB = 8.617333e-5
T_GRID = np.linspace(900.0, 1500.0, 200)
N_EA = 256  # 64 nodes give 0.22 max curve error; 256 + sigma floor at 3x node spacing resolve narrow widths and the burn front (GH quadrature rejected, misses the sharp front)
DTYPE = np.float32

# params (9): [w1, (mu,sig,lnu)x2 families, order n, base]; Nechaev Tables 4/5 fit both orders, Q differs 2x
LO = np.array([0.50, 0.80, 0.05, 0.0, 0.80, 0.05, 0.0, 0.8, 0.00])
HI = np.array([0.97, 4.20, 0.60, 15., 4.20, 0.60, 15., 2.5, 0.30])
NP_ = 9
LOG_BETA_RANGE = (-1.2, 0.2)   # 0.06 - 1.6 K/s (covers 0.17)


def sample_prior(n, rng):
    p = LO + (HI - LO) * rng.random((n, NP_))
    swap = p[:, 1] > p[:, 4]
    for a, b in [(1, 4), (2, 5), (3, 6)]:
        p[swap, a], p[swap, b] = p[swap, b].copy(), p[swap, a].copy()
    return p


def simulate(params, log_beta, noise=0.0, rng=None):
    p = np.atleast_2d(params).astype(DTYPE)
    B = len(p)
    beta = (10.0 ** (np.full(B, log_beta) if np.isscalar(log_beta)
                     else np.asarray(log_beta))).astype(DTYPE)
    Ea = np.linspace(0.5, 5.0, N_EA, dtype=DTYPE)
    Tg = T_GRID.astype(DTYPE)
    dT = DTYPE(Tg[1] - Tg[0])
    boltz = np.exp(-Ea[:, None] / (KB * Tg[None, :]))
    I = np.cumsum(boltz, axis=1) * dT
    r = np.zeros((B, len(Tg)), dtype=DTYPE)
    n_ord = p[:, 7][:, None, None]
    nm1 = np.clip(n_ord - 1.0, 1e-3, None)
    for c in range(2):
        w = p[:, 0] if c == 0 else 1 - p[:, 0]
        mu, sig, lnu = p[:, 1 + 3*c], p[:, 2 + 3*c], p[:, 3 + 3*c]
        nu_b = 10.0 ** lnu / beta
        rho = np.exp(-0.5 * ((Ea[None, :] - mu[:, None]) / sig[:, None]) ** 2)
        rho /= rho.sum(1, keepdims=True)
        x = np.clip(nu_b[:, None, None] * I[None], 0, 80)
        # theta**n in one power op: (1+(n-1)x)^(-n/(n-1)); n->1 limit is exp(-n x)
        theta_n = np.where(np.abs(n_ord - 1.0) < 0.02,
                           np.exp(-n_ord * x),
                           np.power(1.0 + nm1 * x, -n_ord / nm1))
        rate = nu_b[:, None, None] * theta_n * boltz[None]
        r += w[:, None] * (rho[:, :, None] * rate).sum(1)
    r /= r.max(1, keepdims=True) + 1e-12
    r = (r + p[:, 8:9]) / (1 + p[:, 8:9])
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        r = np.clip(r + noise * rng.standard_normal(r.shape).astype(DTYPE), 0, None)
    return r if np.asarray(params).ndim > 1 else r[0]
