"""
Regression tests: a seeded reference-value/determinism check and a negative control for
the low-prefactor artifact finding. Run before any release.
Run: python -m tpdsbi.validation.regression_tests     (~3 min)
"""
import numpy as np, torch
from tpdsbi.analysis.corpus_inversion import CORPUS, load_npe, load_curve, main_family, KB

REFERENCE = {       # from the seed-123 reference run
    "median_Ea_width": 1.23,
    "median_tau_oom": 18.9,
}
TOL = {"median_Ea_width": 0.10, "median_tau_oom": 2.0}


def pilot_headline(seed=123):
    npes = {c: load_npe(c) for c in ["graphene_oxide", "hydrogenated_carbon", "nanodiamond"]}
    ews, tos = [], []
    for name, (cfg_name, csv, lb, refs) in CORPUS.items():
        torch.manual_seed(seed)                      # seeded posterior sampling
        npe = npes[cfg_name]
        S = npe.invert(load_curve(csv, npe.cfg), lb, 4000)
        Ea, ln = main_family(S, cfg_name)
        ews.append(np.percentile(Ea, 95) - np.percentile(Ea, 5))
        tau = np.exp(np.clip(Ea, 0.05, None) / (KB * 298.0)) / 10.0 ** ln
        tos.append(np.log10(np.percentile(tau, 95) / max(np.percentile(tau, 5), 1e-30)))
    return {"median_Ea_width": float(np.median(ews)), "median_tau_oom": float(np.median(tos))}


def check_reference_values():
    print("(1) REFERENCE-VALUE REGRESSION CHECK (seeded sampling)")
    got = pilot_headline()
    ok = True
    for k, v in got.items():
        d = abs(v - REFERENCE[k])
        flag = d <= TOL[k]
        ok &= flag
        print(f"    {k}: {v:.2f} (reference {REFERENCE[k]:.2f}, tol {TOL[k]}) "
              f"{'OK' if flag else '<< REGRESSION'}")
    # determinism: same seed twice -> identical
    got2 = pilot_headline()
    det = all(abs(got[k] - got2[k]) < 1e-9 for k in got)
    print(f"    determinism (same seed twice): {'OK - bit-identical' if det else '<< NON-DETERMINISTIC'}")
    return ok and det


def check_negative_control(n_truth=30):
    print("(2) NEGATIVE CONTROL: single-site ordinary-nu truths must not trigger the artifact")
    import tpdsbi.forward.sim_hydrogenated_carbon as F2
    rng = np.random.default_rng(9)
    npe = load_npe("hydrogenated_carbon")
    nu_in = mono = 0
    for i in range(n_truth):
        Ea0 = rng.uniform(1.2, 2.2)
        # one dominant near-single site (sigma at floor), ordinary nu, first-order
        truth = np.array([[0.9, 0.05, Ea0, 0.05, 13.0, Ea0, 0.05, 13.0, Ea0, 0.05, 13.0,
                           1.0, 0.02]])
        y = np.clip(F2.simulate(truth, np.log10(3.0))[0]
                    + 0.02 * rng.standard_normal(len(F2.T_GRID)).astype(np.float32), 0, None)
        torch.manual_seed(1000 + i)
        S = npe.invert(y, np.log10(3.0), 2000)
        Ea, ln = main_family(S, "hydrogenated_carbon")
        lo, hi = np.percentile(ln, [5, 95])
        nu_in += lo <= 13.0 <= hi                     # true nu inside the CI
        mono += np.percentile(ln, 95) < 8.0           # posterior forces low nu (the artifact)
    print(f"    true nu=1e13 inside 90% CI: {nu_in}/{n_truth} (expected ~{int(0.9*n_truth)})")
    print(f"    posteriors that force nu < 1e8 (artifact): {mono}/{n_truth} (expected 0)")
    return nu_in >= int(0.8 * n_truth) and mono == 0


if __name__ == "__main__":
    ok1 = check_reference_values()
    ok2 = check_negative_control()
    print(f"\nREGRESSION TESTS: {'ALL PASS' if ok1 and ok2 else 'FAILURES - investigate before release'}")
