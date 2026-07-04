"""
Ensemble MCMC comparison: affine-invariant stretch-move sampler (Goodman & Weare) with
split-R-hat and integrated-autocorrelation ESS diagnostics, compared to the flow marginals.
Run: python -m tpdsbi.validation.mcmc_comparison    (~1.5-2 h for both reference curves)
"""
import os, numpy as np
from tpdsbi.analysis.corpus_inversion import CORPUS, load_npe, load_curve, main_family



def stretch_sampler(cfg, y, log_beta, n_walk=64, n_steps=8000, burn=3000,
                    sigma=0.02, a=2.0, seed=0):
    rng = np.random.default_rng(seed)
    X = cfg.sample_prior(n_walk, rng)

    def loglik(Q):
        r = cfg.simulate(Q, log_beta)
        return -0.5 * (((r - y[None, :]) / sigma) ** 2).sum(1)

    ll = loglik(X)
    half = n_walk // 2
    chains = []   # (n_kept, n_walk, NP_)
    acc = tot = 0
    for t in range(n_steps):
        for s0, s1 in [(0, half), (half, n_walk)]:
            idx = slice(s0, s1)
            comp = X[s1 % n_walk: s1 % n_walk + half] if s0 == 0 else X[0:half]
            z = ((a - 1.0) * rng.random(half) + 1.0) ** 2 / a
            j = rng.integers(0, half, half)
            Y = comp[j] + z[:, None] * (X[idx] - comp[j])
            inbox = np.all((Y >= cfg.LO) & (Y <= cfg.HI), 1)
            llY = np.full(half, -np.inf)
            if inbox.any():
                llY[inbox] = loglik(Y[inbox])
            logr = (cfg.NP_ - 1) * np.log(z) + llY - ll[idx]
            accm = np.log(rng.random(half)) < logr
            Xi = X[idx]; Xi[accm] = Y[accm]; X[idx] = Xi
            lli = ll[idx]; lli[accm] = llY[accm]; ll[idx] = lli
            acc += accm.sum(); tot += half
        if t >= burn and t % 5 == 0:
            chains.append(X.copy())
    C = np.array(chains)                      # (n, walkers, d)
    print(f"    acceptance {acc/tot:.2f}, kept {C.shape[0]} x {n_walk} samples")
    return C


def rhat_ess(C):
    """split-R-hat + crude integrated-autocorrelation ESS per parameter.
    C: (n, m, d), n kept steps, m walkers."""
    n, m, d = C.shape
    h = n // 2
    S = np.concatenate([C[:h], C[h:2 * h]], axis=1)     # (h, 2m, d) split chains
    mean_c = S.mean(0)                                   # (2m, d)
    var_c = S.var(0, ddof=1)
    B = h * mean_c.var(0, ddof=1)
    W = var_c.mean(0)
    rhat = np.sqrt(((h - 1) / h * W + B / h) / (W + 1e-12))
    # ESS from ensemble-mean autocorrelation
    ess = np.zeros(d)
    for k in range(d):
        x = C[:, :, k].mean(1)
        x = x - x.mean()
        ac = np.correlate(x, x, "full")[n - 1:] / (np.arange(n, 0, -1) * x.var() + 1e-12)
        tau = 1.0
        for lag in range(1, min(n // 2, 200)):
            if ac[lag] < 0.05:
                break
            tau += 2 * ac[lag]
        ess[k] = n * m / tau
    return rhat, ess


def main():
    print("ENSEMBLE-MCMC COMPARISON (Goodman-Weare stretch move) + diagnostics")
    for name in ["nechaev_hslg", "jung_co2"]:
        cfg_name, csv, lb, _ = CORPUS[name]
        npe = load_npe(cfg_name)
        y = load_curve(csv, npe.cfg)
        print(f"  {name} ...")
        C = stretch_sampler(npe.cfg, y, lb)
        rhat, ess = rhat_ess(C)
        print(f"    split-R-hat: max {rhat.max():.3f} (want < 1.05) | "
              f"min ESS {ess.min():.0f}")
        M = C.reshape(-1, C.shape[2])
        S = npe.invert(y, lb, 8000)
        for q, (Em, Lm), (Ef, Lf) in [("Ea/lnu", main_family(M, cfg_name),
                                       main_family(S, cfg_name))]:
            for lab, aa, bb in [("Ea ", Em, Ef), ("lnu", Lm, Lf)]:
                qm = np.percentile(aa, [5, 50, 95]); qf = np.percentile(bb, [5, 50, 95])
                wr = (qf[2] - qf[0]) / max(qm[2] - qm[0], 1e-9)
                print(f"    {lab}: ensemble {qm[1]:6.2f} [{qm[0]:6.2f},{qm[2]:6.2f}] | "
                      f"flow {qf[1]:6.2f} [{qf[0]:6.2f},{qf[2]:6.2f}] | width ratio {wr:.2f}")
        conv = "converged" if rhat.max() < 1.05 else "not converged (R-hat > 1.05)"
        print(f"    -> {conv}")


if __name__ == "__main__":
    main()
