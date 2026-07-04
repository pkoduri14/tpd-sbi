"""
Forward TPD model for the Jung 2009 (JPC C 113:18480) graphene-oxide reduction case. Free
reaction order n; nth-order Polanyi-Wigner theta(T) = [1+(n-1)(nu/beta)I(T)]^(-1/(n-1)),
I = int exp(-Ea/kT)dT. T grid 300-580 K (their 30-300 C ramp). params (9).
"""
import numpy as np

KB = 8.617333e-5
T_GRID = np.linspace(300.0, 580.0, 200)
N_EA = 256  # 64 nodes give 0.22 max curve error; 256 + sigma floor at 3x node spacing resolve narrow widths and the burn front (GH quadrature rejected, misses the sharp front)

LO = np.array([0.40, 0.80, 0.05, 6.0, 0.80, 0.05, 6.0, 0.8, 0.00])
HI = np.array([0.97, 2.40, 0.25, 20., 2.40, 0.25, 20., 2.5, 0.45])
NP_ = 9
LOG_BETA_RANGE = (-1.0, 0.3)   # 0.1 - 2 K/s (covers Jung's 0.2 / 0.5 / 0.983)


def sample_prior(n, rng):
    p = LO + (HI - LO) * rng.random((n, NP_))
    # sort the two (mu,sig,lnu) blocks by mu
    swap = p[:, 1] > p[:, 4]
    for a, b in [(1, 4), (2, 5), (3, 6)]:
        p[swap, a], p[swap, b] = p[swap, b].copy(), p[swap, a].copy()
    return p


def simulate(params, log_beta, noise=0.0, rng=None):
    p = np.atleast_2d(params).astype(np.float32)
    B = len(p)
    beta = (10.0 ** (np.full(B, log_beta) if np.isscalar(log_beta)
                     else np.asarray(log_beta))).astype(np.float32)
    Ea = np.linspace(0.5, 2.8, N_EA, dtype=np.float32)
    Tg = T_GRID.astype(np.float32)
    dT = np.float32(Tg[1] - Tg[0])
    boltz = np.exp(-Ea[:, None] / (KB * Tg[None, :]))                  # (E,T)
    I = np.cumsum(boltz, axis=1) * dT
    n_ord = p[:, 7][:, None, None]
    r = np.zeros((B, len(Tg)), dtype=np.float32)
    for c in range(2):
        w = p[:, 0] if c == 0 else 1 - p[:, 0]
        mu, sig, lnu = p[:, 1 + 3*c], p[:, 2 + 3*c], p[:, 3 + 3*c]
        nu_b = 10.0 ** lnu / beta
        rho = np.exp(-0.5 * ((Ea[None, :] - mu[:, None]) / sig[:, None]) ** 2)
        rho /= rho.sum(1, keepdims=True)
        x = np.clip(nu_b[:, None, None] * I[None], 0, 80)              # (B,E,T)
        nm1 = np.clip(n_ord - 1.0, 1e-3, None)
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
        r = np.clip(r + noise * rng.standard_normal(r.shape).astype(np.float32), 0, None)
    return r if np.asarray(params).ndim > 1 else r[0]


def peak_T(params, log_beta):
    """Predicted peak temperature(s) at a given heating rate."""
    r = simulate(params, log_beta)
    return T_GRID[np.argmax(r, axis=1)]
