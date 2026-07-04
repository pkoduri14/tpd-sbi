"""
Multi-seed training variance. Trains seeds 1 and 2 for every flow config (seed 0 is the
shipped nets), then recomputes the pilot headline metrics per seed.
Run: python -m tpdsbi.robustness.seed_variance    (~40-60 min CPU)
"""
import os, numpy as np
from tpdsbi.paths import DATA, MODELS, FIGURES
from tpdsbi.inference.neural_posterior import NPE
from tpdsbi.analysis.corpus_inversion import CORPUS, load_npe, load_curve, main_family, KB

SEEDS = [1, 2]
CFGS = ["hydrogenated_carbon", "graphene_oxide", "nanodiamond"]


def pilot_medians(npes):
    ews, tos = [], []
    for name, (cfg_name, csv, lb, refs) in CORPUS.items():
        npe = npes[cfg_name]
        S = npe.invert(load_curve(csv, npe.cfg), lb, 4000)
        Ea, ln = main_family(S, cfg_name)
        ews.append(np.percentile(Ea, 95) - np.percentile(Ea, 5))
        tau = np.exp(np.clip(Ea, 0.05, None) / (KB * 298.0)) / 10.0 ** ln
        tos.append(np.log10(np.percentile(tau, 95) / max(np.percentile(tau, 5), 1e-30)))
    return np.median(ews), np.median(tos)


def main():
    results = {}
    # seed 0 = the shipped nets (flow_<config>.pt)
    npes0 = {c: load_npe(c) for c in CFGS}
    results[0] = pilot_medians(npes0)
    print(f"seed 0 (shipped): median Ea width {results[0][0]:.2f} eV, "
          f"tau spread {results[0][1]:.1f} OoM")

    for seed in SEEDS:
        npes = {}
        for cfg in CFGS:
            npe = NPE(cfg, suffix=f"_s{seed}", n_layers=12)
            if not os.path.exists(os.path.join(MODELS, f"flow_{cfg}_s{seed}.pt")):
                npe.train(n_sims=240000, epochs=40, seed=seed)
            npes[cfg] = npe
        results[seed] = pilot_medians(npes)
        print(f"seed {seed}: median Ea width {results[seed][0]:.2f} eV, "
              f"tau spread {results[seed][1]:.1f} OoM")

    ew = [v[0] for v in results.values()]; to = [v[1] for v in results.values()]
    print("\nMulti-seed variance (3 training seeds):")
    print(f"  median Ea width:  {np.mean(ew):.2f} +/- {np.std(ew):.2f} eV "
          f"(range {min(ew):.2f}-{max(ew):.2f})")
    print(f"  tau298 spread:    {np.mean(to):.1f} +/- {np.std(to):.1f} OoM")


if __name__ == "__main__":
    main()
