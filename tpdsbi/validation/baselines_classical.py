"""
Classical baselines and an MCMC comparison on the same digitized curves: Redhead (1962,
nu=1e13) per curve, Kissinger on Jung's three ramps, and random-walk Metropolis on the
forward model (Gaussian likelihood, sigma=0.02) compared to the flow marginals.
Run: python baselines.py      (~5 min; MCMC is the slow part)
"""
import os, numpy as np
from tpdsbi.paths import DATA, MODELS, FIGURES
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tpdsbi.analysis.corpus_inversion import CORPUS, load_npe, load_curve, main_family, KB



def redhead(Tm, beta, nu=1e13):
    return KB * Tm * (np.log(nu * Tm / beta) - 3.64)


def kissinger(Tms, betas):
    """slope of ln(Tm^2/beta) vs 1/Tm = Ea/kB (first-order)."""
    x = 1.0 / np.asarray(Tms)
    y = np.log(np.asarray(Tms) ** 2 / np.asarray(betas))
    return np.polyfit(x, y, 1)[0] * KB


def mcmc(cfg, y, log_beta, n_walk=64, n_steps=12000, burn=5000, sigma=0.02, seed=0):
    rng = np.random.default_rng(seed)
    P = cfg.sample_prior(n_walk, rng)
    def loglik(Q):
        r = cfg.simulate(Q, log_beta)
        return -0.5 * (((r - y[None, :]) / sigma) ** 2).sum(1)
    ll = loglik(P)
    scale = 0.004 * (cfg.HI - cfg.LO)   # tuned: ~0.2 acceptance
    chain = []
    acc = 0
    for t in range(n_steps):
        Q = P + scale * rng.standard_normal(P.shape)
        inbox = np.all((Q >= cfg.LO) & (Q <= cfg.HI), 1)
        llq = np.full(n_walk, -np.inf)
        if inbox.any():
            llq[inbox] = loglik(Q[inbox])
        a = np.log(rng.random(n_walk)) < (llq - ll)
        P[a] = Q[a]; ll[a] = llq[a]; acc += a.mean()
        if t >= burn and t % 10 == 0:
            chain.append(P.copy())
    print(f"    MCMC acceptance {acc/n_steps:.2f}, {len(chain)*n_walk} samples")
    return np.vstack(chain)


def main():
    npes = {c: load_npe(c) for c in ["graphene_oxide", "hydrogenated_carbon", "nanodiamond"]}

    # ---------- (a) Redhead on every curve ----------
    print("(a) REDHEAD baseline (nu=1e13 assumed), vs our calibrated posterior:")
    print(f"  {'curve':13s} {'Tm(K)':>6s} {'Redhead Ea':>10s} {'posterior Ea 90% CI':>22s} "
          f"{'Redhead inside?':>15s}")
    for name, (cfg_name, csv, lb, refs) in CORPUS.items():
        npe = npes[cfg_name]
        y = load_curve(csv, npe.cfg)
        Tm = npe.cfg.T_GRID[np.argmax(y)]
        rEa = redhead(Tm, 10.0 ** lb)
        S = npe.invert(y, lb, 4000)
        Ea, _ = main_family(S, cfg_name)
        lo, hi = np.percentile(Ea, [5, 95])
        print(f"  {name:13s} {Tm:6.0f} {rEa:8.2f} eV      [{lo:5.2f}, {hi:5.2f}] eV "
              f"{'inside' if lo <= rEa <= hi else 'OUTSIDE':>15s}")

    # ---------- (b) Kissinger implementation check (Jung) ----------
    Ea_k = kissinger([392.9, 402.1, 407.3], [0.2, 0.5, 0.983])
    print(f"\n(b) KISSINGER on Jung's three ramps: Ea = {Ea_k:.2f} eV "
          f"(their published: 1.39 +/- 0.17 -> {'REPRODUCED' if abs(Ea_k-1.39)<0.17 else 'mismatch'})")

    # ---------- (c) MCMC reference vs flow ----------
    print("\n(c) MCMC posterior vs flow (main-family marginals):")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for row, name in enumerate(["nechaev_hslg", "jung_co2"]):
        cfg_name, csv, lb, _ = CORPUS[name]
        npe = npes[cfg_name]
        y = load_curve(csv, npe.cfg)
        print(f"  {name} ...")
        M = mcmc(npe.cfg, y, lb)
        S = npe.invert(y, lb, 8000)
        for col, q in enumerate(["Ea", "lnu"]):
            Em, Lm = main_family(M, cfg_name)
            Ef, Lf = main_family(S, cfg_name)
            a, b = (Em, Ef) if q == "Ea" else (Lm, Lf)
            qm = np.percentile(a, [5, 50, 95]); qf = np.percentile(b, [5, 50, 95])
            wr = (qf[2] - qf[0]) / max(qm[2] - qm[0], 1e-9)
            print(f"    {q:3s}: MCMC {qm[1]:6.2f} [{qm[0]:6.2f},{qm[2]:6.2f}] | "
                  f"flow {qf[1]:6.2f} [{qf[0]:6.2f},{qf[2]:6.2f}] | width ratio {wr:.2f}")
            ax = axes[row, col]
            ax.hist(a, bins=50, alpha=0.5, density=True, label="MCMC (reference)")
            ax.hist(b, bins=50, alpha=0.5, density=True, label="flow (26 ms)")
            ax.set_title(f"{name} - main-family {q}", fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "baseline_mcmc.png"), dpi=130)
    print("  wrote figures/baseline_mcmc.png")


if __name__ == "__main__":
    main()
