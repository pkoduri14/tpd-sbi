"""
Validate the posterior on Jung et al. 2009 (JPC C 113:18480), where the same GO-reduction
was measured by TPD-Kissinger (1.39+/-0.17 eV) and independent transport (1.60 eV). Invert
the digitized TPD curve plus two extra heating rates; check the posterior against both.
Run: python -m tpdsbi.validation.validate_ground_truth    (~10 min; ABC-SMC, no network)
"""
import os, numpy as np
from tpdsbi.paths import DATA, MODELS, FIGURES
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tpdsbi.forward.sim_graphene_oxide import simulate, peak_T, sample_prior, T_GRID, LO, HI, KB
from tpdsbi.forward.batch_simulate import sim_chunked, peakT_chunked

LB30 = np.log10(0.5)                    # 30 C/min
RAMPS_TM = [(np.log10(0.2), 392.9), (np.log10(0.983), 407.3)]   # (log10 beta, Tm_obs K)
EA_TPD, EA_TPD_SD = 1.39, 0.17          # their Kissinger result (eV)
EA_TRANSPORT, EA_TR_SD = 1.60, 0.04     # their independent transport result (eV)


def load_obs():
    d = np.loadtxt(os.path.join(DATA, "jung2009_go_co2.csv"), delimiter=",", skiprows=1)
    T, V = d[:, 0], d[:, 1]
    V = V / V.max()
    y = np.interp(T_GRID, T, V, left=V[:10].mean(), right=V[-10:].mean())
    return np.clip(y, 0, None)


def dist(P, y, use_tm):
    d = np.sqrt(((sim_chunked(simulate, P, LB30) - y[None, :]) ** 2).mean(1))
    if use_tm:
        for lb, tm in RAMPS_TM:
            d += 0.02 * np.abs(peakT_chunked(peak_T, P, lb) - tm) / 3.0
    return d


def abc_smc(y, use_tm, rng, n_keep=500, rounds=5, n0=200000, n_round=120000):
    P = sample_prior(n0, rng)
    d = dist(P, y, use_tm)
    k = np.argsort(d)[:n_keep]; P, d = P[k], d[k]
    print(f"    round 0: eps={d.max():.4f}")
    for r in range(1, rounds):
        idx = rng.integers(0, len(P), n_round)
        scale = (HI - LO) * 0.06 / r
        Q = np.clip(P[idx] + scale * rng.standard_normal((n_round, len(LO))), LO + 1e-6, HI)
        swap = Q[:, 1] > Q[:, 4]
        for a, b in [(1, 4), (2, 5), (3, 6)]:
            Q[swap, a], Q[swap, b] = Q[swap, b].copy(), Q[swap, a].copy()
        dq = dist(Q, y, use_tm)
        allP = np.vstack([P, Q]); alld = np.concatenate([d, dq])
        k = np.argsort(alld)[:n_keep]; P, d = allP[k], alld[k]
        print(f"    round {r}: eps={d.max():.4f} best={d.min():.4f}")
    return P


def dominant_Ea(P):
    """Ea of the family responsible for the main peak (higher weight)."""
    w1 = P[:, 0]
    return np.where(w1 >= 0.5, P[:, 1], P[:, 4]), np.where(w1 >= 0.5, P[:, 3], P[:, 6])


def report(P, label):
    Ea, lnu = dominant_Ea(P)
    q = np.percentile(Ea, [5, 50, 95]); k = np.percentile(lnu, [5, 50, 95])
    n = np.percentile(P[:, 7], [5, 50, 95])
    inT = q[0] <= EA_TPD <= q[2]; inTr = q[0] <= EA_TRANSPORT <= q[2]
    print(f"  {label}")
    print(f"    main-peak Ea: {q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}] eV "
          f"(width {q[2]-q[0]:.2f} eV)")
    print(f"      covers TPD-Kissinger 1.39+/-0.17? {'YES' if inT else 'no'} | "
          f"covers transport 1.60+/-0.04? {'YES' if inTr else 'no'}")
    print(f"    log10 nu: {k[1]:.1f} [{k[0]:.1f}, {k[2]:.1f}]  | "
          f"order n: {n[1]:.2f} [{n[0]:.2f}, {n[2]:.2f}]")
    return Ea, lnu


def main():
    rng = np.random.default_rng(0)
    y = load_obs()
    print("JUNG 2009 GO-REDUCTION VALIDATION (independent ground truth)")
    print("  [A] single ramp (30 C/min curve only):")
    PA = abc_smc(y, False, rng)
    EaA, lnuA = report(PA, "posterior A (single ramp):")
    print("  [B] + Jung's other two heating rates (Tm @ 12 & 59 C/min):")
    PB = abc_smc(y, True, rng)
    EaB, lnuB = report(PB, "posterior B (three ramps):")

    shrink = (np.percentile(EaA, 95) - np.percentile(EaA, 5)) / \
             (np.percentile(EaB, 95) - np.percentile(EaB, 5) + 1e-9)
    print(f"\n  multi-ramp gain on real data: Ea 90%-interval shrinks {shrink:.1f}x")

    # GO shelf life at 25 C from posterior B (GO is known to age/reduce over weeks-months at RT)
    tau = np.exp(np.clip(EaB, 0.1, None) / (KB * 298.0)) / 10.0 ** lnuB
    lo, med, hi = np.percentile(tau, [5, 50, 95])
    print(f"  implied GO shelf life @25C (1/k of main channel): "
          f"median {_h(med)}, 90% band [{_h(lo)} .. {_h(hi)}]")

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.6))
    a = ax[0, 0]
    a.plot(T_GRID, y, "k.", ms=3, label="Jung Fig.6a CO2 (digitized)")
    pp = simulate(PB[rng.choice(len(PB), 80)], LB30)
    a.plot(T_GRID, pp.T, color="C0", alpha=0.06)
    a.plot([], [], "C0", label="posterior predictive (3-ramp)")
    a.set_xlabel("T (K)"); a.set_ylabel("CO2 rate (norm.)")
    a.set_title("GO-reduction TPD: posterior-predictive fit"); a.legend(fontsize=8)

    a = ax[0, 1]
    for Ea, lnu, col, lab in [(EaA, lnuA, "C3", "single ramp"), (EaB, lnuB, "C0", "3 ramps")]:
        a.scatter(Ea, lnu, s=4, alpha=0.3, color=col, label=lab)
    a.axvspan(EA_TPD - EA_TPD_SD, EA_TPD + EA_TPD_SD, color="green", alpha=0.15,
              label="published Kissinger 1.39±0.17")
    a.axvspan(EA_TRANSPORT - EA_TR_SD, EA_TRANSPORT + EA_TR_SD, color="orange", alpha=0.3,
              label="published transport 1.60±0.04")
    a.set_xlabel("main-peak Ea (eV)"); a.set_ylabel("log10 nu")
    a.set_title("Ea posterior and published values"); a.legend(fontsize=7)

    a = ax[1, 0]
    a.hist(EaA, bins=50, alpha=0.5, color="C3", density=True, label="single ramp")
    a.hist(EaB, bins=50, alpha=0.5, color="C0", density=True, label="3 ramps")
    a.axvline(EA_TPD, color="green", ls="--", label="TPD-Kissinger")
    a.axvline(EA_TRANSPORT, color="orange", ls="--", label="transport (independent)")
    a.set_xlabel("main-peak Ea (eV)"); a.set_ylabel("posterior density")
    a.set_title("Posterior with one vs three heating rates"); a.legend(fontsize=7)

    a = ax[1, 1]
    a.hist(PA[:, 7], bins=40, alpha=0.5, color="C3", density=True, label="single ramp")
    a.hist(PB[:, 7], bins=40, alpha=0.5, color="C0", density=True, label="3 ramps")
    a.axvline(2.0, color="k", ls="--", label="published resistivity: n~2")
    a.set_xlabel("reaction order n"); a.set_ylabel("posterior density")
    a.set_title("Reaction-order posterior"); a.legend(fontsize=8)

    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "jung_validation.png"), dpi=140)
    print(f"\n  wrote figures/jung_validation.png")
    np.save(os.path.join(MODELS, "abc_posterior_jung.npy"), PB)


def _h(s):
    for u, v in [("yr", 3.15e7), ("d", 86400), ("h", 3600), ("m", 60)]:
        if s >= v: return f"{s/v:.1f}{u}"
    return f"{s:.1f}s"


if __name__ == "__main__":
    main()
