"""
Calibrated Bayesian inversion of a single TPD/TDS curve from a two-column CSV (T_K, signal).

    python analyze_my_tpd.py --demo
    python analyze_my_tpd.py mydata.csv --beta 0.5 --hold 298 --out results/mine

Output: a 4-panel figure + posterior_samples.csv in --out (default figures/user/).
"""
import argparse, os, sys
from tpdsbi.paths import DATA, MODELS, FIGURES
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB = 8.617333e-5
CONFIGS = [   # (module, T_low, T_high, description)
    ("graphene_oxide", 300.0, 580.0, "low-T decomposition (GO-class, order inferred)"),
    ("hydrogenated_carbon",   250.0, 900.0, "mid-T desorption (hydrogenated-carbon class, 3 families)"),
    ("nanodiamond", 900.0, 1500.0, "high-T desorption (graphitic/nanodiamond class)"),
]
ORDER_IDX = {"hydrogenated_carbon": 11, "graphene_oxide": 7, "nanodiamond": 7}


def pick_config(Tmin, Tmax):
    best, cover = None, -1
    for name, lo, hi, desc in CONFIGS:
        c = max(0.0, min(Tmax, hi) - max(Tmin, lo)) / max(Tmax - Tmin, 1e-9)
        if c > cover:
            best, cover = (name, lo, hi, desc), c
    return best, cover


def main():
    ap = argparse.ArgumentParser(description="Calibrated Bayesian inversion of your TPD curve")
    ap.add_argument("csv", nargs="?", help="two-column CSV: T_K, signal")
    ap.add_argument("--beta", type=float, help="heating rate in K/s")
    ap.add_argument("--hold", type=float, default=298.0, help="retention hold temperature (K)")
    ap.add_argument("--out", default=os.path.join(FIGURES, "user"))
    ap.add_argument("--demo", action="store_true",
                    help="run on a shipped example (Jung 2009 CO2, beta=0.5 K/s)")
    a = ap.parse_args()

    if a.demo:
        a.csv = os.path.join(DATA, "jung2009_go_co2.csv"); a.beta = 0.5
        print("[demo] using the shipped Jung 2009 graphene-oxide CO2 curve, beta = 0.5 K/s\n")
    if not a.csv or a.beta is None:
        ap.error("provide a CSV and --beta (or use --demo)")

    # ---- 1. load + preprocess ----------------------------------------------------------
    d = np.loadtxt(a.csv, delimiter=",", skiprows=1)
    T, V = d[:, 0], d[:, 1]
    print(f"[1] loaded {len(T)} points, T = {T.min():.0f}-{T.max():.0f} K, "
          f"peak at {T[np.argmax(V)]:.0f} K")
    (cfg_name, lo, hi, desc), cover = pick_config(T.min(), T.max())
    print(f"[2] model family: {cfg_name} ({desc}); grid {lo:.0f}-{hi:.0f} K "
          f"covers {100*cover:.0f}% of your data" +
          ("" if cover > 0.85 else "  << WARNING: partial coverage"))

    from tpdsbi.inference.neural_posterior import NPE
    npe = NPE(cfg_name, n_layers=12)
    cfg = npe.cfg
    V = V / V.max()
    base = float(V[:10].mean()); right = float(V[-10:].mean())
    y = np.interp(cfg.T_GRID, T, V, left=base, right=right)
    if right > 3 * base:   # truncated mid-descent: continue the measured slope
        sl = np.polyfit(T[-25:], V[-25:], 1)[0]
        if sl < 0:
            ext = cfg.T_GRID > T[-1]
            y[ext] = np.clip(right + sl * (cfg.T_GRID[ext] - T[-1]), min(base, 0.05), None)
            print("    (curve ends mid-descent; tail extended along the measured slope)")
    y = np.clip(y, 0, None)
    lb = np.log10(a.beta)
    if not (cfg.LOG_BETA_RANGE[0] <= lb <= cfg.LOG_BETA_RANGE[1]):
        print(f"    << WARNING: beta={a.beta} K/s is outside the trained range "
              f"{10**cfg.LOG_BETA_RANGE[0]:.2f}-{10**cfg.LOG_BETA_RANGE[1]:.2f} K/s; "
              f"results are an extrapolation")

    # ---- 3. invert -----------------------------------------------------------------------
    S = npe.invert(y, lb, 6000)
    print(f"[3] inverted: 6000 posterior samples (~30 ms)")

    # ---- 4. fit check --------------------------------------------------------------------
    rng = np.random.default_rng(0)
    pp = cfg.simulate(S[rng.choice(len(S), 150)], lb)
    rmse = float(np.sqrt(((pp.mean(0) - y) ** 2).mean()))
    ok = rmse < 0.12
    print(f"[4] posterior-predictive fit RMSE = {rmse:.3f} "
          + ("(good fit)" if ok else "<< WARNING: poor fit; curve may be outside the model class"))

    # ---- report parameters ---------------------------------------------------------------
    print("[5] posterior summaries (median [90% credible interval]):")
    nfam = 3 if cfg_name == "hydrogenated_carbon" else 2
    w = (np.column_stack([S[:, 0], S[:, 1], 1 - S[:, 0] - S[:, 1]]) if nfam == 3
         else np.column_stack([S[:, 0], 1 - S[:, 0]]))
    off = 2 if cfg_name == "hydrogenated_carbon" else 1
    for f in range(nfam):
        Ea, sg, ln = S[:, off + 3*f], S[:, off + 1 + 3*f], S[:, off + 2 + 3*f]
        print(f"    family {f+1} (weight {np.median(w[:, f]):.2f}): "
              f"Ea = {np.median(Ea):.2f} [{np.percentile(Ea,5):.2f}, {np.percentile(Ea,95):.2f}] eV | "
              f"site-width {np.median(sg):.2f} eV | "
              f"log10 nu = {np.median(ln):.1f} [{np.percentile(ln,5):.1f}, {np.percentile(ln,95):.1f}]")
    n = S[:, ORDER_IDX[cfg_name]]
    print(f"    reaction order n = {np.median(n):.2f} [{np.percentile(n,5):.2f}, "
          f"{np.percentile(n,95):.2f}]")

    # ---- 5. retention forecast ------------------------------------------------------------
    # dominant family's per-draw tau at the hold temperature
    fdom = np.argmax(np.median(w, 0))
    Ea, ln = S[:, off + 3*fdom], S[:, off + 2 + 3*fdom]
    tau = np.exp(np.clip(Ea, 0.05, None) / (KB * a.hold)) / 10.0 ** ln
    lo_t, med_t, hi_t = np.percentile(tau, [5, 50, 95])
    def h(s):
        for u, v in [("yr", 3.15e7), ("d", 86400), ("h", 3600), ("min", 60)]:
            if s >= v: return f"{s/v:.1f} {u}"
        return f"{s:.2g} s"
    print(f"[6] retention at {a.hold:.0f} K (dominant family 1/k): "
          f"median {h(med_t)}, 90% band [{h(lo_t)} .. {h(hi_t)}] "
          f"({np.log10(max(hi_t,1e-30)/max(lo_t,1e-30)):.1f} orders of magnitude)")
    print("[7] to narrow this: (a) rerun the same sample at a 2nd heating rate;")
    print("    (b) an isothermal hold (fixed T, watch the signal decay) measures the")
    print("    retention curve directly; (c) an initial-coverage series separates site")
    print("    heterogeneity from coverage-dependent kinetics.")

    # ---- outputs ---------------------------------------------------------------------------
    os.makedirs(a.out, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].plot(cfg.T_GRID, y, "k.", ms=3, label="your curve")
    ax[0, 0].plot(cfg.T_GRID, pp[:60].T, color="C0", alpha=0.06)
    ax[0, 0].plot([], [], "C0", label="posterior predictive")
    ax[0, 0].set_xlabel("T (K)"); ax[0, 0].set_ylabel("rate (norm.)")
    ax[0, 0].set_title(f"Fit (RMSE {rmse:.3f})"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].scatter(Ea[::4], ln[::4], s=3, alpha=0.25)
    ax[0, 1].set_xlabel("dominant-family Ea (eV)"); ax[0, 1].set_ylabel("log10 nu")
    ax[0, 1].set_title("Joint posterior of (Ea, log10 nu)")
    Eg = np.linspace(cfg.LO[off], cfg.HI[off], 200)
    dens = np.zeros_like(Eg)
    for i in rng.choice(len(S), 400):
        for f in range(nfam):
            dens += w[i, f] * np.exp(-0.5*((Eg - S[i, off+3*f]) / S[i, off+1+3*f])**2) \
                    / S[i, off+1+3*f]
    ax[1, 0].plot(Eg, dens / dens.max())
    ax[1, 0].set_xlabel("Ea (eV)"); ax[1, 0].set_ylabel("site density (norm.)")
    ax[1, 0].set_title("Posterior-mean binding-energy distribution")
    times = np.logspace(0, 10, 200)
    frac = np.array([(tau[:, None] > times[None, :]).mean(0)]).ravel()
    ax[1, 1].semilogx(times, frac, "C0")
    q_lo = np.array([(np.percentile(tau, 5) > t) for t in times]).astype(float)
    ax[1, 1].set_xlabel(f"time at {a.hold:.0f} K (s)")
    ax[1, 1].set_ylabel("P(dominant site still occupied)")
    ax[1, 1].set_title("Retention forecast (posterior-averaged)")
    fig.tight_layout()
    fp = os.path.join(a.out, "analysis.png")
    fig.savefig(fp, dpi=140)
    np.savetxt(os.path.join(a.out, "posterior_samples.csv"), S, delimiter=",",
               header="see model config for column order", comments="")
    print(f"[8] wrote {fp} and posterior_samples.csv")


if __name__ == "__main__":
    main()
