"""
Evaluate the trained NPE on a synthetic carbon TPD curve built from published Nechaev
2020 values, and show how the Ea-nu compensation ridge widens room-temperature retention.
Run: python -m tpdsbi.validation.validate_synthetic   (needs npe.pt from demo_mdn.py)
Output: figures/npe_inversion.png
"""
import os, numpy as np, torch
from tpdsbi.paths import DATA, MODELS, FIGURES
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tpdsbi.forward.sim_twopeak_demo import simulate, retention_curve, T_GRID, PNAMES, KB
from tpdsbi.inference.demo_mdn import MDN


# ground truth sample: Nechaev-like carbon, weak 0.46 eV + strong 0.88 eV families, prefactor 10^3 (measured range 1e1-1e6)
TRUTH = np.array([0.55, 0.46, 0.07, 0.88, 0.10, 3.0])


def main():
    rng = np.random.default_rng(7)
    model = MDN(); model.load_state_dict(torch.load(os.path.join(MODELS, "npe.pt"))); model.eval()

    y_obs = simulate(TRUTH[None, :], noise=0.02, rng=rng)[0]
    S = model.sample(torch.tensor(y_obs[None, :], dtype=torch.float32), 4000)

    print("NPE inversion of a Nechaev-grounded carbon TPD curve")
    print(f"  {'param':8s} {'truth':>7s} {'posterior median [5%, 95%]':>30s}  in CI?")
    for i, k in enumerate(PNAMES):
        lo, med, hi = np.percentile(S[:, i], [5, 50, 95])
        print(f"  {k:8s} {TRUTH[i]:7.2f}   {med:7.2f}  [{lo:6.2f}, {hi:6.2f}]   "
              f"{'YES' if lo <= TRUTH[i] <= hi else 'no'}")

    # ---- compensation ridge ----
    # posterior tau at 298 K for the weak family
    tau_RT = np.exp(np.clip(S[:, 1], 0.05, None) / (KB * 298.0)) / 10.0 ** S[:, 5]
    lo_t, med_t, hi_t = np.percentile(tau_RT, [5, 50, 95])
    print(f"\n  weak-family retention @298 K:")
    print(f"    posterior median {_h(med_t)}   90% band [{_h(lo_t)} .. {_h(hi_t)}]")
    print(f"    band width: {np.log10(hi_t/lo_t):.1f} orders of magnitude")

    # correlation along the ridge
    r = np.corrcoef(S[:, 1], S[:, 5])[0, 1]
    print(f"\n  compensation ridge: corr(mu1, log_nu) = {r:+.2f}")

    # ---- figure: 4 panels ----
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    a = ax[0, 0]
    a.plot(T_GRID, y_obs, "k.", ms=3, label="measured (noisy)")
    post_curves = simulate(S[rng.choice(len(S), 60)], noise=0.0)
    a.plot(T_GRID, post_curves.T, color="C0", alpha=0.08)
    a.plot([], [], color="C0", label="posterior predictive")
    a.set_xlabel("T (K)"); a.set_ylabel("desorption rate (norm.)")
    a.set_title("TPD curve + posterior predictive fit"); a.legend(fontsize=8)

    a = ax[0, 1]
    a.scatter(S[:, 1], S[:, 5], s=3, alpha=0.2, color="C0")
    a.scatter([TRUTH[1]], [TRUTH[5]], marker="*", s=220, color="red", edgecolor="k",
              zorder=5, label="truth")
    a.set_xlabel("weak-family Ea (eV)"); a.set_ylabel("log10 nu")
    a.set_title("Joint posterior of weak-family (Ea, log10 nu)")
    a.legend(fontsize=8)

    a = ax[1, 0]
    Ea_grid = np.linspace(0.2, 1.4, 200)
    dens = np.zeros_like(Ea_grid)
    for w1, m1, s1, m2, s2, _ in S[rng.choice(len(S), 400)]:
        dens += (w1 * np.exp(-0.5 * ((Ea_grid - m1) / s1) ** 2) / s1
                 + (1 - w1) * np.exp(-0.5 * ((Ea_grid - m2) / s2) ** 2) / s2)
    tw1, tm1, ts1, tm2, ts2, _ = TRUTH
    tdens = (tw1 * np.exp(-0.5 * ((Ea_grid - tm1) / ts1) ** 2) / ts1
             + (1 - tw1) * np.exp(-0.5 * ((Ea_grid - tm2) / ts2) ** 2) / ts2)
    a.plot(Ea_grid, dens / dens.max(), "C0", label="posterior mean rho(Ea)")
    a.plot(Ea_grid, tdens / tdens.max(), "k--", label="truth")
    a.set_xlabel("Ea (eV)"); a.set_ylabel("site density (norm.)")
    a.set_title("Posterior binding-energy distribution"); a.legend(fontsize=8)

    a = ax[1, 1]
    times = np.logspace(0, 10, 150)
    for p in S[rng.choice(len(S), 80)]:
        _, f = retention_curve(p, times=times, n_mc=1500, rng=rng)
        a.semilogx(times, f, color="C0", alpha=0.06)
    _, ft = retention_curve(TRUTH, times=times, n_mc=4000, rng=rng)
    a.semilogx(times, ft, "k--", lw=2, label="truth")
    a.plot([], [], color="C0", label="posterior band")
    a.axvline(86400, ls=":", c="gray"); a.text(86400*1.3, 0.9, "1 day", fontsize=7)
    a.axvline(90*86400, ls=":", c="gray"); a.text(90*86400*1.3, 0.9, "season", fontsize=7)
    a.set_xlabel("time at 298 K (s)"); a.set_ylabel("fraction H retained")
    a.set_title("Retention forecast (90% band)"); a.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "npe_inversion.png"), dpi=140)
    print(f"\n  wrote figures/npe_inversion.png")


def _h(s):
    for u, v in [("yr", 3.15e7), ("d", 86400), ("h", 3600), ("m", 60)]:
        if s >= v: return f"{s/v:.1f}{u}"
    return f"{s:.1f}s"


if __name__ == "__main__":
    main()
