"""
Tool-chain checks for the forward simulator: analytic peak position (Polanyi-Wigner),
numerics convergence vs a float64/N_EA=256 reference, and digitization-jitter sensitivity.
Run: python -m tpdsbi.validation.verify_numerics   (~3 min)
"""
import os, numpy as np
from tpdsbi.paths import DATA, MODELS, FIGURES
from scipy.optimize import brentq
import tpdsbi.forward.sim_hydrogenated_carbon as F2
from tpdsbi.analysis.corpus_inversion import CORPUS, load_npe, load_curve, main_family

KB = 8.617333e-5


# ---------- (1) analytic peak-position check ----------
def analytic_tm(Ea, nu, beta):
    f = lambda T: Ea / (KB * T * T) - (nu / beta) * np.exp(-Ea / (KB * T))
    return brentq(f, 260, 2000)


def check_analytic():
    print("(1) ANALYTIC Tm CHECK (single-site, first-order, vs Polanyi-Wigner)")
    dT = F2.T_GRID[1] - F2.T_GRID[0]
    worst = 0.0
    for Ea in [0.9, 1.3, 1.8, 2.3]:
        for lnu in [6.0, 10.0, 13.0]:
            for beta in [0.17, 0.5, 3.0]:
                try:
                    tm_true = analytic_tm(Ea, 10.0 ** lnu, beta)
                except ValueError:
                    continue
                if not (F2.T_GRID[5] < tm_true < F2.T_GRID[-5]):
                    continue
                p = np.array([[0.44, Ea, 0.02, lnu, Ea, 0.02, lnu, Ea, 0.02, lnu,
                               1.0, 0.0]])
                # 13-param layout: [w1,w2,3x(mu,sig,lnu),order,base]
                p = np.array([[0.44, 0.44, Ea, 0.02, lnu, Ea, 0.02, lnu, Ea, 0.02, lnu,
                               1.0, 0.0]])
                y = F2.simulate(p, np.log10(beta))[0]
                tm_sim = F2.T_GRID[int(np.argmax(y))]
                worst = max(worst, abs(tm_sim - tm_true))
    print(f"    worst |Tm_sim - Tm_analytic| = {worst:.1f} K "
          f"(T-grid spacing {dT:.1f} K) -> {'PASS' if worst <= 2 * dT else 'FAIL'}")


# ---------- (2) numerics convergence ----------
def check_numerics():
    print("(2) NUMERICS CONVERGENCE (quantified, vs float64 / N_EA=256 reference)")
    rng = np.random.default_rng(3)
    P = F2.sample_prior(200, rng)
    lb = rng.uniform(*F2.LOG_BETA_RANGE, 200)
    base = F2.simulate(P, lb)
    # reference: crank N_EA and float64
    NE0, DT0 = F2.N_EA, F2.DTYPE
    F2.N_EA, F2.DTYPE = 256, np.float64
    ref = F2.simulate(P, lb)
    F2.N_EA, F2.DTYPE = NE0, DT0
    err = np.abs(base - ref).max()
    print(f"    max |curve(N_EA=64, f32) - curve(N_EA=256, f64)| = {err:.4f} "
          f"(noise floor 0.02) -> {'PASS' if err < 0.01 else 'FAIL'}")


# ---------- (3) digitization-jitter sensitivity ----------
def check_jitter():
    print("(3) DIGITIZATION-JITTER SENSITIVITY (T shift +/-3K, T scale +/-1%, y +/-3%)")
    rng = np.random.default_rng(7)
    npes = {c: load_npe(c) for c in ["graphene_oxide", "hydrogenated_carbon", "nanodiamond"]}
    print(f"    {'curve':13s} {'base Ea med/width':>18s} {'jitter spread of med':>21s} "
          f"{'width change':>13s}")
    for name in ["jung_co2", "nechaev_hslg", "fig3a"]:
        cfg_name, csv, lb, _ = CORPUS[name]
        npe = npes[cfg_name]
        d = np.loadtxt(os.path.join(DATA, csv), delimiter=",", skiprows=1)
        meds, wids = [], []
        for k in range(9):
            T, V = d[:, 0].copy(), d[:, 1].copy()
            if k > 0:
                T = T * (1 + rng.uniform(-0.01, 0.01)) + rng.uniform(-3, 3)
                V = V * (1 + rng.uniform(-0.03, 0.03))
            V = V / V.max()
            b = float(V[:10].mean()); r = float(V[-10:].mean())
            y = np.interp(npe.cfg.T_GRID, T, V, left=b, right=r)
            if r > 3 * b:
                sl = np.polyfit(T[-25:], V[-25:], 1)[0]
                if sl < 0:
                    ext = npe.cfg.T_GRID > T[-1]
                    y[ext] = np.clip(r + sl * (npe.cfg.T_GRID[ext] - T[-1]),
                                     min(b, 0.05), None)
            S = npe.invert(np.clip(y, 0, None), lb, 3000)
            Ea, _ = main_family(S, cfg_name)
            meds.append(np.median(Ea))
            wids.append(np.percentile(Ea, 95) - np.percentile(Ea, 5))
        print(f"    {name:13s} {meds[0]:8.2f} / {wids[0]:4.2f} eV "
              f"{np.std(meds[1:]):17.3f} eV {np.std(wids[1:]):10.3f} eV")


if __name__ == "__main__":
    check_analytic()
    check_numerics()
    check_jitter()
