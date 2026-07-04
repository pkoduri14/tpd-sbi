"""
Forward TPD model for inverting Nechaev 2020 (IJHE 45:25030) Fig. 4: hydrogenated epitaxial
graphene, D2 signal, beta = 3 K/s. Three site families with per-family prefactor, free
reaction order, heating-rate conditioning, baseline nuisance. params (13).
"""
import numpy as np

KB = 8.617333e-5
T_GRID = np.linspace(250.0, 900.0, 200)
N_EA = 256  # 64 nodes give 0.22 max curve error; 256 + sigma floor at 3x node spacing resolve narrow widths and the burn front (GH quadrature rejected, misses the sharp front)
DTYPE = np.float32  # float32 throughout: ~4x faster, 2x less memory, noise floor 2% anyway

# params (13): [w1, w2, (mu,sig,lnu)x3 families, order n, base]
# per-family theta^n assumes quasi-independent families (cross-family recombination neglected)
LO = np.array([0.05, 0.05, 0.25, 0.05, 0.0, 0.25, 0.05, 0.0, 0.25, 0.05, 0.0, 0.8, 0.0])
HI = np.array([0.90, 0.90, 2.20, 0.25, 14., 2.20, 0.25, 14., 2.20, 0.25, 14., 2.5, 0.15])
NP_ = 13
LOG_BETA_RANGE = (-1.0, 0.7)   # beta 0.1 - 5 K/s (covers 0.17 and 3.0)


def sample_prior(n, rng):
    p = LO + (HI - LO) * rng.random((n, NP_))
    # weights: renormalize (w1,w2,w3) to sum 1
    w3 = rng.uniform(0.05, 0.90, n)
    s = p[:, 0] + p[:, 1] + w3
    p[:, 0] /= s; p[:, 1] /= s
    # sort the three (mu, sig, lnu) blocks by mu for identifiability
    blocks = p[:, 2:11].reshape(n, 3, 3)
    order = np.argsort(blocks[:, :, 0], axis=1)
    p[:, 2:11] = np.take_along_axis(blocks, order[:, :, None], axis=1).reshape(n, 9)
    return p


def simulate(params, log_beta, noise=0.0, rng=None):
    """params (B,13), log_beta scalar or (B,) log10 of K/s -> normalized curves (B,200)."""
    p = np.atleast_2d(params).astype(DTYPE)
    B = len(p)
    beta = (10.0 ** (np.full(B, log_beta) if np.isscalar(log_beta)
                     else np.asarray(log_beta))).astype(DTYPE)
    Ea = np.linspace(0.15, 2.6, N_EA, dtype=DTYPE)
    dT = DTYPE(T_GRID[1] - T_GRID[0])
    Tg = T_GRID.astype(DTYPE)
    boltz = np.exp(-Ea[:, None] / (KB * Tg[None, :]))              # (E, T)
    I = np.cumsum(boltz, axis=1) * dT                              # (E, T)
    r = np.zeros((B, len(T_GRID)), dtype=DTYPE)
    w = np.column_stack([p[:, 0], p[:, 1], 1 - p[:, 0] - p[:, 1]])
    n_ord = p[:, 11][:, None, None]
    nm1 = np.clip(n_ord - 1.0, 1e-3, None)
    for c in range(3):
        mu, sig, lnu = p[:, 2 + 3*c], p[:, 3 + 3*c], p[:, 4 + 3*c]
        nu_b = (10.0 ** lnu / beta)                                 # (B,)
        rho = np.exp(-0.5 * ((Ea[None, :] - mu[:, None]) / sig[:, None]) ** 2)
        rho /= rho.sum(1, keepdims=True)                            # (B, E)
        x = np.clip(nu_b[:, None, None] * I[None], 0, 80)           # (B,E,T)
        # theta**n in one power op: (1+(n-1)x)^(-n/(n-1)); n->1 limit is exp(-n x)
        theta_n = np.where(np.abs(n_ord - 1.0) < 0.02,
                           np.exp(-n_ord * x),
                           np.power(1.0 + nm1 * x, -n_ord / nm1))
        rate = nu_b[:, None, None] * theta_n * boltz[None]
        r += w[:, c, None] * (rho[:, :, None] * rate).sum(1)
    r /= r.max(1, keepdims=True) + 1e-12
    r = (r + p[:, 12:13]) / (1 + p[:, 12:13])                       # baseline
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        r = np.clip(r + noise * rng.standard_normal(r.shape).astype(DTYPE), 0, None)
    return r if np.asarray(params).ndim > 1 else r[0]


def retention_samples(P, T_hold=298.0, times=None, rng=None):
    """Posterior retention curves: P (n,12) -> (n, len(times)) fraction retained at T_hold."""
    rng = rng or np.random.default_rng(0)
    times = np.logspace(0, 10, 150) if times is None else times
    out = np.zeros((len(P), len(times)))
    for i, p in enumerate(P):
        w = np.array([p[0], p[1], 1 - p[0] - p[1]])
        n_mc = 2000
        parts = []
        for c in range(3):
            k = max(1, int(n_mc * max(w[c], 0)))
            Ea = rng.normal(p[2 + 3*c], p[3 + 3*c], k)
            parts.append(np.exp(np.clip(Ea, 0.05, None) / (KB * T_hold)) / 10.0 ** p[4 + 3*c])
        tau = np.concatenate(parts)
        out[i] = [(tau > t).mean() for t in times]
    return times, out
