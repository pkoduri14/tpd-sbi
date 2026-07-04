"""
Sensitivity studies: instrument-effect and rival-mechanism checks, each asking whether a
known unmodeled effect changes the conclusions.
Run: python -m tpdsbi.robustness.sensitivity_studies   (~30 min if the lateral_interaction flow needs training)
"""
import os, numpy as np
from tpdsbi.paths import DATA, MODELS, FIGURES
from tpdsbi.inference.neural_posterior import NPE
from tpdsbi.analysis.corpus_inversion import CORPUS, load_npe, load_curve, main_family

KB = 8.617333e-5


# ---------------- lateral-interaction mechanism demo ----------------
def lateral_mechanism_demo():
    import tpdsbi.forward.sim_lateral_interaction as FL
    print("lateral-interaction truth, single-site fits (E0=2.0 eV, nu=1e13, first-order)")
    print("    single-site FIXED-Ea fit (the literature's model) of each curve:")
    lb = np.log10(3.0)
    for al in [0.0, 0.3, 0.6, 1.0, 1.5]:
        y = FL.simulate(np.array([[2.0, al, 13.0, 0.02]]), lb)[0]
        best = (1e9, 0, 0)
        eas = np.linspace(0.4, 2.6, 90); lns = np.linspace(0.0, 14.0, 90)
        E, L = np.meshgrid(eas, lns)
        P = np.column_stack([E.ravel(), np.zeros(E.size), L.ravel(),
                             np.full(E.size, 0.02)])
        X = np.vstack([FL.simulate(P[i:i+4000], lb) for i in range(0, len(P), 4000)])
        d = np.sqrt(((X - y[None, :]) ** 2).mean(1))
        i = int(np.argmin(d))
        print(f"    alpha={al:4.1f} eV/ML: fitted (Ea={P[i,0]:.2f} eV, K0=1e{P[i,2]:.1f}) "
              f"RMSE {d[i]:.3f}"
              + ("   <- truth recovered (no broadening)" if al == 0 else ""))


# ---------------- rival-model fit of the real curves ----------------
def lateral_vs_distribution():
    print("rival model (lateral interactions) vs site distribution on real curves")
    lat = NPE("lateral_interaction", n_layers=8)
    if not os.path.exists(os.path.join(MODELS, "flow_lateral_interaction.pt")):
        lat.train(n_sims=120000, epochs=30, seed=0)
    dist = load_npe("hydrogenated_carbon")
    rng = np.random.default_rng(0)
    for name in ["nechaev_hslg", "nechaev_flg"]:
        cfg_name, csv, lb, _ = CORPUS[name]
        yl = load_curve(csv, lat.cfg)
        yd = load_curve(csv, dist.cfg)
        Sl = lat.invert(yl, lb, 4000)
        Sd = dist.invert(yd, lb, 4000)
        ppl = lat.cfg.simulate(Sl[rng.choice(len(Sl), 150)], lb)
        ppd = dist.cfg.simulate(Sd[rng.choice(len(Sd), 150)], lb)
        rl = np.sqrt(((ppl.mean(0) - yl) ** 2).mean())
        rd = np.sqrt(((ppd.mean(0) - yd) ** 2).mean())
        q = np.percentile(Sl[:, 1], [5, 50, 95])
        print(f"    {name}: lateral-model RMSE {rl:.3f} (alpha={q[1]:.2f} [{q[0]:.2f},{q[2]:.2f}])"
              f" | site-distribution RMSE {rd:.3f}"
              f" -> {'indistinguishable (state both)' if abs(rl-rd) < 0.02 else ('distribution preferred' if rd < rl else 'lateral preferred')}")


# ---------------- instrument lag + smearing ----------------
def instrument_lag_smear():
    print("thermocouple lag + detector smearing (beta = 3 K/s curves)")
    npe = load_npe("hydrogenated_carbon")
    for name in ["nechaev_hslg", "nechaev_flg"]:
        cfg_name, csv, lb, _ = CORPUS[name]
        d = np.loadtxt(os.path.join(DATA, csv), delimiter=",", skiprows=1)
        base_S = npe.invert(load_curve(csv, npe.cfg), lb, 3000)
        Ea0, _ = main_family(base_S, "hydrogenated_carbon")
        m0 = np.median(Ea0)
        outs = []
        for tau in [1.0, 2.0, 3.0]:                     # seconds of thermal lag
            dd = d.copy(); dd[:, 0] = dd[:, 0] - tau * 3.0
            np.savetxt(os.path.join(DATA, "_tmp_lag.csv"), dd, delimiter=",",
                       header="T_K,s", comments="")
            S = npe.invert(load_curve("_tmp_lag.csv", npe.cfg), lb, 3000)
            Ea, _ = main_family(S, "hydrogenated_carbon")
            outs.append(f"lag {tau:.0f}s: dEa {np.median(Ea)-m0:+.3f}")
        # smearing: convolve signal with exponential kernel sigma_T = beta*tau_d
        for taud in [1.0, 2.0]:
            sig_T = 3.0 * taud
            dT = np.median(np.diff(d[:, 0]))
            k = int(max(1, round(3 * sig_T / dT)))
            ker = np.exp(-np.arange(0, 3 * k) * dT / sig_T); ker /= ker.sum()
            dd = d.copy(); dd[:, 1] = np.convolve(d[:, 1], ker, mode="same")
            np.savetxt(os.path.join(DATA, "_tmp_lag.csv"), dd, delimiter=",",
                       header="T_K,s", comments="")
            S = npe.invert(load_curve("_tmp_lag.csv", npe.cfg), lb, 3000)
            Ea, _ = main_family(S, "hydrogenated_carbon")
            outs.append(f"smear {taud:.0f}s: dEa {np.median(Ea)-m0:+.3f}")
        print(f"    {name}: " + " | ".join(outs) + " eV")
    os.remove(os.path.join(DATA, "_tmp_lag.csv"))


# ---------------- correlated noise ----------------
def correlated_noise():
    print("correlated noise: AR(1) phi=0.9 + slow drift (coverage, 150 sims/config)")
    rng = np.random.default_rng(31)
    for cfg_name in ["graphene_oxide", "hydrogenated_carbon", "nanodiamond"]:
        npe = NPE(cfg_name, n_layers=12)
        cfg = npe.cfg
        P = cfg.sample_prior(150, rng)
        lb = rng.uniform(*cfg.LOG_BETA_RANGE, 150)
        clean = cfg.simulate(P, lb)
        # AR(1) scaled to 2% marginal sd + linear drift +/-2%
        e = rng.standard_normal(clean.shape).astype(clean.dtype)
        ar = np.zeros_like(e)
        for t in range(1, e.shape[1]):
            ar[:, t] = 0.9 * ar[:, t - 1] + e[:, t]
        ar *= 0.02 / ar.std()
        drift = np.linspace(0, 1, clean.shape[1])[None, :] * \
            rng.uniform(-0.02, 0.02, (len(P), 1)).astype(clean.dtype)
        X = np.clip(clean + ar + drift, 0, None)
        cov = np.zeros(cfg.NP_)
        for i in range(len(X)):
            s = npe.invert(X[i], lb[i], 600)
            lo, hi = np.percentile(s, [5, 95], axis=0)
            cov += (P[i] >= lo) & (P[i] <= hi)
        print(f"    {cfg_name}: 90% coverage under AR(1)+drift = {cov.mean()/len(X):.2f}")


# ---------------- readsorption ----------------
def readsorption_bias():
    print("readsorption (constant strength R): analytically rate -> rate/(1+R), i.e.")
    print("    nu_apparent = nu/(1+R); Ea unbiased. Numerical confirmation, R=3:")
    import tpdsbi.forward.sim_hydrogenated_carbon as F2
    rng = np.random.default_rng(41)
    npe = load_npe("hydrogenated_carbon")
    truth = np.array([[0.6, 0.3, 1.0, 0.1, 8.0, 1.6, 0.1, 12.0, 2.0, 0.1, 13.0, 1.0, 0.02]])
    y = F2.simulate(truth, np.log10(3.0))
    # readsorption R=3: nu -> nu/4 for every family = lnu - log10(4)
    tr = truth.copy(); tr[0, [4, 7, 10]] -= np.log10(4)
    yR = F2.simulate(tr, np.log10(3.0))
    S = npe.invert(np.clip(yR[0] + 0.02 * rng.standard_normal(200).astype(np.float32), 0, None),
                   np.log10(3.0), 3000)
    Ea, ln = main_family(S, "hydrogenated_carbon")
    print(f"    truth family-1 (Ea=1.0, lnu=8.0), observed through R=3 readsorption:")
    print(f"    posterior Ea {np.median(Ea):.2f} [{np.percentile(Ea,5):.2f},{np.percentile(Ea,95):.2f}]"
          f" | lnu {np.median(ln):.1f} (expect ~{8-np.log10(4):.1f} = lnu - log10(1+R))")


# ---------------- diffusion-limited rate law ----------------
def diffusion_limited_fit():
    print("diffusion-limited (slab out-diffusion, 15-mode sum) grid-fit on jung_co2")
    import tpdsbi.forward.sim_graphene_oxide as FG
    y = load_curve("jung2009_go_co2.csv", FG)
    Tg = FG.T_GRID.astype(np.float64)
    dT = Tg[1] - Tg[0]
    beta = 0.5
    lamr = (2 * np.arange(1, 16) - 1) ** 2          # mode rate ratios 1, 9, 25, ...
    w_n = (8 / np.pi ** 2) / lamr                    # initial mode weights
    best = (1e9, 0, 0)
    for ED in np.linspace(0.6, 2.2, 80):
        boltz = np.exp(-ED / (KB * Tg))
        I = np.cumsum(boltz) * dT
        for lk in np.linspace(-2, 12, 120):          # log10 (lambda1^2 D0 / L^2)
            k1b = 10.0 ** lk / beta
            rate = np.zeros_like(Tg)
            for lr, wn in zip(lamr, w_n):
                x = np.clip(k1b * lr * I, 0, 80)
                rate += wn * k1b * lr * np.exp(-x) * boltz
            rate /= rate.max() + 1e-12
            rate = (rate + 0.30) / 1.30              # match observed baseline fraction
            d = np.sqrt(((rate - y) ** 2).mean())
            if d < best[0]:
                best = (d, ED, lk)
    print(f"    best diffusion fit: E_D={best[1]:.2f} eV, log10(k1/s)={best[2]:.1f}, "
          f"RMSE {best[0]:.3f} (desorption-model RMSE ~0.08)")


# ---------------- coupled-coverage recombination ----------------
def coupled_recombination_bias():
    print("coupled-coverage recombination: truth couples families via theta_total;")
    print("    inverted with the independent-families flow (bias check):")
    import tpdsbi.forward.sim_hydrogenated_carbon as F2
    rng = np.random.default_rng(52)
    npe = load_npe("hydrogenated_carbon")
    Tg = F2.T_GRID.astype(np.float64)
    dT = Tg[1] - Tg[0]
    hits = 0
    for trial in range(4):
        Eas = np.sort(rng.uniform(0.9, 2.0, 3))
        lnus = rng.uniform(10, 13, 3)
        ws = rng.dirichlet([2, 2, 2])
        th = ws.copy().astype(np.float64)
        rate = np.zeros_like(Tg)
        for t, T in enumerate(Tg):                   # coupled: rate_f ∝ theta_f * theta_tot
            tt = th.sum()
            k = 10.0 ** lnus / 0.5 * np.exp(-Eas / (KB * T))
            d = np.clip(k * th * tt * dT, 0, th)
            th -= d
            rate[t] = d.sum()
        y = rate / (rate.max() + 1e-12)
        y = np.clip((y + 0.02) / 1.02
                    + 0.02 * rng.standard_normal(len(y)), 0, None).astype(np.float32)
        S = npe.invert(y, np.log10(3.0), 3000)
        Ea, _ = main_family(S, "hydrogenated_carbon")
        dom = Eas[np.argmax(ws)]
        lo, hi = np.percentile(Ea, [5, 95])
        hits += lo <= dom <= hi
    print(f"    dominant-family Ea inside 90% CI in {hits}/4 coupled-truth trials")


if __name__ == "__main__":
    lateral_mechanism_demo(); lateral_vs_distribution(); instrument_lag_smear(); correlated_noise(); readsorption_bias(); diffusion_limited_fit(); coupled_recombination_bias()
