"""
Additional validation: retention (pushforward) coverage, noise-model robustness of the
real-corpus headline, and the lateral-interaction integrator convergence audit.
Run: python -m tpdsbi.validation.pushforward_and_noise     (~2 h; the three retrainings dominate)
"""
import os, numpy as np
from tpdsbi.paths import DATA, MODELS, FIGURES
from tpdsbi.inference.neural_posterior import NPE
from tpdsbi.analysis.corpus_inversion import CORPUS, load_npe, load_curve, main_family, KB



def push_tau(S, cfg_name):
    Ea, ln = main_family(S, cfg_name)
    return np.log10(np.exp(np.clip(Ea, 0.05, None) / (KB * 298.0)) / 10.0 ** ln)


def pushforward_coverage():
    print("PUSHFORWARD (retention) COVERAGE (joint-posterior functional)")
    rng = np.random.default_rng(71)
    for cfg_name in ["hydrogenated_carbon", "nanodiamond"]:
        npe = load_npe(cfg_name) if cfg_name == "hydrogenated_carbon" else NPE(cfg_name, n_layers=12)
        cfg = npe.cfg
        N = 300
        P = cfg.sample_prior(N, rng)
        lb = rng.uniform(*cfg.LOG_BETA_RANGE, N)
        X = cfg.simulate(P, lb, noise=0.02, rng=rng)
        c50 = c90 = 0
        for i in range(N):
            S = npe.invert(X[i], lb[i], 800)
            tp = push_tau(S, cfg_name)
            tt = push_tau(P[i:i + 1], cfg_name)[0]
            lo, hi = np.percentile(tp, [5, 95]); c90 += lo <= tt <= hi
            lo, hi = np.percentile(tp, [25, 75]); c50 += lo <= tt <= hi
        print(f"   {cfg_name}: log10-tau(298K) coverage 50% -> {c50/N:.2f}, 90% -> {c90/N:.2f}"
              f"   {'PASS' if abs(c90/N - 0.9) < 0.05 and abs(c50/N - 0.5) < 0.07 else 'FLAG'}")


def noise_marginalized_headline():
    print("NOISE-MARGINALIZED FLOWS (sigma ~ U(0.005, 0.05), unknown to the network)")
    metrics = {}
    for tag, suffix in [("fixed-sigma (paper nets)", ""), ("noise-marginalized", "_nm")]:
        npes = {}
        for cfg in ["hydrogenated_carbon", "graphene_oxide", "nanodiamond"]:
            if suffix == "":
                npes[cfg] = load_npe(cfg)
            else:
                npe = NPE(cfg, suffix="_nm", n_layers=12)
                if not os.path.exists(os.path.join(MODELS, f"flow_{cfg}_nm.pt")):
                    npe.train(n_sims=240000, epochs=40, seed=0, noise=(0.005, 0.05))
                npes[cfg] = npe
        ews, tos = [], []
        for name, (cfg_name, csv, lb, refs) in CORPUS.items():
            npe = npes[cfg_name]
            S = npe.invert(load_curve(csv, npe.cfg), lb, 4000)
            Ea, ln = main_family(S, cfg_name)
            ews.append(np.percentile(Ea, 95) - np.percentile(Ea, 5))
            tau = np.exp(np.clip(Ea, 0.05, None) / (KB * 298.0)) / 10.0 ** ln
            tos.append(np.log10(np.percentile(tau, 95) / max(np.percentile(tau, 5), 1e-30)))
        metrics[tag] = (np.median(ews), np.median(tos))
        print(f"   {tag:26s}: median Ea width {metrics[tag][0]:.2f} eV, "
              f"tau spread {metrics[tag][1]:.1f} OoM")
    d = abs(metrics["fixed-sigma (paper nets)"][0] - metrics["noise-marginalized"][0])
    print(f"   headline shift under unknown-noise treatment: {d:.2f} eV "
          f"-> {'ROBUST to unknown noise level' if d < 0.15 else 'SENSITIVE - report'}")


def lateral_integrator_audit():
    print("LATERAL-INTERACTION INTEGRATOR CONVERGENCE AUDIT (production SUB vs SUB=512/f64)")
    import tpdsbi.forward.sim_lateral_interaction as FL
    rng = np.random.default_rng(73)
    P = FL.sample_prior(300, rng)
    lb = rng.uniform(*FL.LOG_BETA_RANGE, 300)
    base = FL.simulate(P, lb)
    S0, D0 = FL.SUB, FL.DTYPE
    FL.SUB, FL.DTYPE = 512, np.float64
    ref = FL.simulate(P, lb)
    FL.SUB, FL.DTYPE = S0, D0
    err = np.abs(base - ref)
    rms = np.sqrt((err ** 2).mean(1))
    print(f"   max abs {err.max():.4f} | worst-curve RMS {rms.max():.4f} | "
          f"median RMS {np.median(rms):.5f}  "
          f"{'PASS' if rms.max() < 0.02 else 'FLAG (tighten integrator)'}")


if __name__ == "__main__":
    lateral_integrator_audit()
    pushforward_coverage()
    noise_marginalized_headline()
