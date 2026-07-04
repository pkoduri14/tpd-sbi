"""
Calibration checks for the trained flows: multi-level coverage, SBC rank uniformity,
and a joint log-density rank test.
Run: python -m tpdsbi.validation.calibration     (~10 min CPU for all three flows)
Output: figures/sbc_coverage.png
"""
import os, numpy as np, torch
from tpdsbi.paths import DATA, MODELS, FIGURES
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import chisquare, kstest
from tpdsbi.inference.neural_posterior import Flow, NPE

LEVELS = [0.5, 0.68, 0.8, 0.9, 0.95]
N_TEST = 400
N_POST = 500


def load(cfg_name):
    return NPE(cfg_name, n_layers=12)


def log_q(npe, x_t, P):
    """flow log-density at params P, including the logit-transform jacobian."""
    q = np.clip((P - npe.cfg.LO) / (npe.cfg.HI - npe.cfg.LO), 1e-5, 1 - 1e-5)
    z = np.log(q / (1 - q))
    with torch.no_grad():
        lp = npe.model.log_prob(x_t.expand(len(P), -1),
                                torch.tensor(z, dtype=torch.float32)).numpy()
    ljac = -np.log(q * (1 - q) * (npe.cfg.HI - npe.cfg.LO)).sum(1)
    return lp + ljac


def main():
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for row, cfg_name in enumerate(["hydrogenated_carbon", "graphene_oxide", "nanodiamond"]):
        npe = load(cfg_name)
        cfg = npe.cfg
        P = cfg.sample_prior(N_TEST, rng)
        lb = rng.uniform(*cfg.LOG_BETA_RANGE, N_TEST)
        X = cfg.simulate(P, lb, noise=0.02, rng=rng)

        ranks = np.zeros((N_TEST, cfg.NP_))
        jranks = np.zeros(N_TEST)
        cov = {lv: np.zeros(cfg.NP_) for lv in LEVELS}
        for i in range(N_TEST):
            S = npe.invert(X[i], lb[i], N_POST)
            ranks[i] = (S < P[i][None, :]).mean(0)
            for lv in LEVELS:
                a = (1 - lv) / 2 * 100
                lo, hi = np.percentile(S, [a, 100 - a], axis=0)
                cov[lv] += (P[i] >= lo) & (P[i] <= hi)
            x_t = torch.tensor(np.append(X[i], lb[i])[None, :], dtype=torch.float32)
            lq_true = log_q(npe, x_t, P[i][None, :])[0]
            lq_samp = log_q(npe, x_t, S)
            # randomized rank (breaks ties/discretization; uniform under calibration)
            k = (lq_samp < lq_true).sum()
            jranks[i] = (k + rng.random()) / (len(S) + 1)

        # coverage curves
        print(f"[{cfg_name}] multi-level coverage (nominal -> empirical mean over params):")
        emp = []
        for lv in LEVELS:
            e = cov[lv].mean() / N_TEST
            emp.append(e)
            print(f"   {lv:.0%} -> {e:.2f}  (per-param range "
                  f"{cov[lv].min()/N_TEST:.2f}-{cov[lv].max()/N_TEST:.2f})")
        ax = axes[row, 0]
        for j in range(cfg.NP_):
            ax.plot(LEVELS, [cov[lv][j] / N_TEST for lv in LEVELS], "o-", alpha=0.3, ms=3)
        ax.plot([0.4, 1], [0.4, 1], "k--"); ax.set_xlabel("nominal"); ax.set_ylabel("empirical")
        ax.set_title(f"{cfg_name}: coverage curves (per param)", fontsize=9)

        # SBC rank uniformity, chi-square on 20 bins per parameter
        ax = axes[row, 1]
        n_fail = 0
        for j in range(cfg.NP_):
            h, _ = np.histogram(ranks[:, j], bins=20, range=(0, 1))
            p = chisquare(h).pvalue
            n_fail += p < 0.01
            ax.hist(ranks[:, j], bins=20, range=(0, 1), histtype="step", alpha=0.5)
        ax.axhline(N_TEST / 20, color="k", ls="--")
        ax.set_title(f"{cfg_name}: SBC rank histograms (per param)", fontsize=9)
        print(f"   SBC uniformity: {n_fail}/{cfg.NP_} parameters fail at p<0.01")

        # joint log-density rank
        ax = axes[row, 2]
        ax.hist(jranks, bins=20, range=(0, 1))
        ax.axhline(N_TEST / 20, color="k", ls="--")
        hj, _ = np.histogram(jranks, bins=20, range=(0, 1))
        pj = chisquare(hj).pvalue
        ax.set_title(f"{cfg_name}: joint log-density rank", fontsize=9)
        print(f"   joint calibration: KS uniformity p = {pj:.3f} "
              f"({'ok' if pj > 0.01 else 'MISCALIBRATED'})")

    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "sbc_coverage.png"), dpi=130)
    print("\nwrote figures/sbc_coverage.png")


if __name__ == "__main__":
    main()
