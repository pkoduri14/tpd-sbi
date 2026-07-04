"""
Reproduction driver: runs the analysis suite from the digitized corpus and shipped nets.

    python reproduce.py            # core suite (~2 h)
    python reproduce.py --full     # + slow validations and robustness studies (~12 h)
    python reproduce.py --retrain  # retrain all flows first (~2 h extra)
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = []
if "--retrain" in sys.argv:
    STEPS += [("train flows (all families, order-n, audited sim)",
               ["python", "-m", "tpdsbi.inference.neural_posterior", "hydrogenated_carbon", "graphene_oxide", "nanodiamond"])]
STEPS += [
    ("numerics + digitization-jitter verification",     ["python", "-m", "tpdsbi.validation.verify_numerics"]),
    ("calibration: multi-level coverage + SBC + joint", ["python", "-m", "tpdsbi.validation.calibration"]),
    ("validation: simulated ground truth (6p demo)",    ["python", "-m", "tpdsbi.validation.validate_synthetic"]),
    ("pilot: 10-curve inversion + headline metrics",    ["python", "-m", "tpdsbi.analysis.corpus_inversion"]),
    ("prior contraction + sensitivity",                 ["python", "-m", "tpdsbi.analysis.prior_sensitivity"]),
    ("mechanism demo (low-prefactor artifact)",         ["python", "-m", "tpdsbi.analysis.low_prefactor_artifact"]),
    ("baselines: Redhead/Kissinger/MCMC",               ["python", "-m", "tpdsbi.validation.baselines_classical"]),
]
if "--full" in sys.argv:
    STEPS += [
        ("validation: Jung ground truth (ABC-SMC, slow)", ["python", "-m", "tpdsbi.validation.validate_ground_truth"]),
        ("ablations (noise/size/families/point-NN)",      ["python", "-m", "tpdsbi.robustness.ablations"]),
        ("must-fixes (pushforward, noise-marg, lat)",     ["python", "-m", "tpdsbi.validation.pushforward_and_noise"]),
        ("ensemble MCMC + diagnostics",                   ["python", "-m", "tpdsbi.validation.mcmc_comparison"]),
        ("multi-seed variance",                           ["python", "-m", "tpdsbi.robustness.seed_variance"]),
        ("per-family reaction-order robustness",          ["python", "-m", "tpdsbi.robustness.reaction_order_test"]),
        ("sensitivity studies (instrument/mechanism)",    ["python", "-m", "tpdsbi.robustness.sensitivity_studies"]),
    ]

if __name__ == "__main__":
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for label, cmd in STEPS:
        print(f"\n{'='*70}\n{label}\n{'='*70}", flush=True)
        r = subprocess.run(cmd, cwd=HERE, env=env)
        if r.returncode != 0:
            sys.exit(f"step failed: {label}")
    print("\nAll steps reproduced. Figures in figures/.")
