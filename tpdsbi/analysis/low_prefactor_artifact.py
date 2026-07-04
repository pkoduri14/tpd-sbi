"""
Simulate first-order desorption with nu=1e13 on a heterogeneous surface (site width sigma),
then fit it with a single-site first-order model and record the reported (Ea, K0) vs sigma.
Overlays Nechaev 2020 Table-4 K0 values (9-180 s^-1) on the fitted trajectory.

Run: python -m tpdsbi.analysis.low_prefactor_artifact    (~2 min)
"""
import os, numpy as np
from tpdsbi.paths import DATA, MODELS, FIGURES
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tpdsbi.forward.sim_nanodiamond import simulate, T_GRID, KB

LB = np.log10(0.17)                    # Nechaev's heating rate
EA_TRUE, LNU_TRUE = 2.4, 13.0          # ordinary physics
SIGMAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
# Nechaev Table 4 (1st-order single-site fits): (Q kJ/mol -> eV, K0 s^-1)
NECHAEV = [("YDA", 97/96.485, 9.0), ("Fe", 111/96.485, 60.0),
           ("Ni", 117/96.485, 100.0), ("Co", 122/96.485, 180.0)]


def hetero_curve(sig):
    """single heterogeneous family: both families identical -> one distribution."""
    p = np.array([[0.5, EA_TRUE, sig, LNU_TRUE, EA_TRUE, sig, LNU_TRUE, 1.0, 0.02]])
    return simulate(p, LB)[0]


def single_site_fit(y, tol=0.005):
    """Fit a single-site (sigma at prior floor) first-order model. Returns the best-fit
    (Ea, K0), its RMSE, and the full near-optimal set of (Ea, K0) within tol of the best."""
    eas = np.linspace(0.4, 4.0, 120); lns = np.linspace(0.0, 15.0, 120)
    E, L = np.meshgrid(eas, lns)
    P = np.column_stack([np.full(E.size, 0.5), E.ravel(), np.full(E.size, 0.03),
                         L.ravel(), E.ravel(), np.full(E.size, 0.03), L.ravel(),
                         np.ones(E.size), np.full(E.size, 0.02)])
    X = np.vstack([simulate(P[i:i+8000], LB) for i in range(0, len(P), 8000)])
    d = np.sqrt(((X - y[None, :]) ** 2).mean(1))
    i = int(np.argmin(d))
    acc = d < d[i] + tol
    return (P[i, 1], P[i, 3], d[i], P[acc, 1], P[acc, 3])


def main():
    print("single-site fits of heterogeneous first-order desorption")
    print(f"  truth: Ea = {EA_TRUE} eV, nu = 1e13 s^-1, first-order, site-distribution width sigma")
    print(f"  {'sigma (eV)':>10s} {'best-fit Ea':>11s} {'best K0':>10s} {'RMSE':>6s} "
          f"{'acceptable-set Ea range':>24s} {'MAX K0 in set':>13s}")
    fits = []
    for sig in SIGMAS:
        y = hetero_curve(sig)
        ea, ln, rmse, eaA, lnA = single_site_fit(y)
        fits.append((sig, ea, ln, rmse, eaA, lnA))
        print(f"  {sig:10.2f} {ea:9.2f} eV {10.0**ln:10.1e} {rmse:6.3f} "
              f"{eaA.min():10.2f}-{eaA.max():4.2f} eV {10.0**lnA.max():13.1e}")

    sig_a, ea_a, ln_a = np.array([f[0] for f in fits]), np.array([f[1] for f in fits]), \
                        np.array([f[2] for f in fits])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    a = axes[0]
    for sig, c in [(0.05, "C0"), (0.25, "C1"), (0.50, "C3")]:
        a.plot(T_GRID, hetero_curve(sig), c, label=f"sigma={sig} eV")
    a.set_xlabel("T (K)"); a.set_ylabel("rate (norm.)"); a.legend(fontsize=8)
    a.set_title("Simulated spectra vs site-width sigma\n(Ea=2.4 eV, nu=1e13, 1st order)", fontsize=9)

    a = axes[1]
    a.plot(sig_a, 10.0 ** ln_a, "o-")
    a.set_yscale("log"); a.axhline(1e13, ls="--", c="k")
    a.text(0.06, 2e13, "true nu", fontsize=8)
    for name, ea, k0 in NECHAEV:
        a.axhline(k0, ls=":", c="C3", alpha=0.6)
    a.text(0.35, 30, "Nechaev K0 values", fontsize=8, color="C3")
    a.set_xlabel("true site-distribution width sigma (eV)")
    a.set_ylabel("single-site fitted K0 (1/s)")
    a.set_title("Best single-site fitted K0\nvs true site width", fontsize=9)

    a = axes[2]
    a.plot(ea_a, ln_a, "o-", label="single-site fits (sigma 0.05-0.5)")
    a.scatter([EA_TRUE], [LNU_TRUE], marker="*", s=250, c="k", label="truth")
    for name, ea, k0 in NECHAEV:
        a.scatter([ea], [np.log10(k0)], marker="s", s=60, c="C3")
        a.annotate(name, (ea, np.log10(k0)), fontsize=7, xytext=(4, 4),
                   textcoords="offset points")
    a.set_xlabel("fitted Ea (eV)"); a.set_ylabel("fitted log10 K0")
    a.set_title("Single-site fits of broadened truths\nand published values", fontsize=9)
    a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "low_prefactor_artifact.png"), dpi=140)
    print("\n  wrote figures/low_prefactor_artifact.png")


if __name__ == "__main__":
    main()
