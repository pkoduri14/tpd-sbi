"""
Forward TPD simulator: Polanyi-Wigner desorption from a heterogeneous surface. rho(Ea) is a
2-Gaussian site-family mixture sharing one prefactor nu; linear ramp T = T0 + beta*t, first
order dtheta_i/dT = -(nu/beta) theta_i exp(-Ea/(kB*T)) integrated over rho_i(Ea). Output is
the normalized desorption-rate curve r(T). Generates training data for the posterior estimator.
"""
import numpy as np

KB = 8.617333e-5  # eV/K

# instrument temperature grid (typical carbon TPD range)
T_GRID = np.linspace(150.0, 1000.0, 200)
BETA = 0.5          # K/s heating rate
N_EA = 120          # quadrature nodes over Ea

# prior ranges for the 2-Gaussian mixture; spans Nechaev's measured peaks 0.34-0.88 eV and the atomic-H tail
PRIOR = {
    "w1":      (0.15, 0.85),   # weight of family 1
    "mu1":     (0.30, 0.80),   # eV
    "sig1":    (0.02, 0.15),   # eV, width (disorder) of family 1
    "mu2":     (0.80, 1.60),   # eV
    "sig2":    (0.02, 0.20),   # eV
    "log_nu":  (2.0, 15.0),    # log10 prefactor: spans measured 1e2 (Nechaev) to 1e15
}
PNAMES = list(PRIOR.keys())


def sample_prior(n, rng):
    lo = np.array([PRIOR[k][0] for k in PNAMES]); hi = np.array([PRIOR[k][1] for k in PNAMES])
    return lo + (hi - lo) * rng.random((n, len(PNAMES)))


def simulate(params, noise=0.0, rng=None):
    """params: (..., 6) array [w1, mu1, sig1, mu2, sig2, log_nu] -> rate curves (..., len(T_GRID))."""
    p = np.atleast_2d(params)
    w1, mu1, s1, mu2, s2, lnu = (p[:, i] for i in range(6))
    nu = 10.0 ** lnu
    # quadrature over Ea for each mixture
    Ea = np.linspace(0.1, 2.2, N_EA)                                   # (E,)
    rho = (w1[:, None] * np.exp(-0.5 * ((Ea - mu1[:, None]) / s1[:, None]) ** 2) / s1[:, None]
           + (1 - w1[:, None]) * np.exp(-0.5 * ((Ea - mu2[:, None]) / s2[:, None]) ** 2) / s2[:, None])
    rho /= rho.sum(1, keepdims=True)                                    # (B, E) site fractions
    # first-order theta(T) = exp(-(nu/beta) I(T,Ea)), I = int exp(-Ea/kB T')dT' cumulative on the T grid
    dT = T_GRID[1] - T_GRID[0]
    boltz = np.exp(-Ea[None, :, None] / (KB * T_GRID[None, None, :]))   # (1, E, T)
    I = np.cumsum(boltz, axis=2) * dT                                   # (1, E, T)
    theta = np.exp(-(nu[:, None, None] / BETA) * I)                     # (B, E, T)
    rate = (nu[:, None, None] / BETA) * theta * boltz                   # (B, E, T)
    r = (rho[:, :, None] * rate).sum(1)                                 # (B, T)
    r /= r.max(1, keepdims=True) + 1e-12                                # normalize like real data
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        r = np.clip(r + noise * rng.standard_normal(r.shape), 0, None)
    return r if params.ndim > 1 else r[0]


def retention_curve(params, T_hold=298.0, times=None, n_mc=4000, rng=None):
    """Fraction of H retained vs time at T_hold for one param set; sites with
    tau(Ea) = exp(Ea/kB T)/nu shorter than t have leaked."""
    rng = rng or np.random.default_rng(0)
    w1, mu1, s1, mu2, s2, lnu = params
    n1 = int(n_mc * w1)
    Ea = np.concatenate([rng.normal(mu1, s1, n1), rng.normal(mu2, s2, n_mc - n1)])
    tau = np.exp(np.clip(Ea, 0.05, None) / (KB * T_hold)) / 10.0 ** lnu
    times = np.logspace(0, 9, 200) if times is None else times
    return times, np.array([(tau > t).mean() for t in times])
