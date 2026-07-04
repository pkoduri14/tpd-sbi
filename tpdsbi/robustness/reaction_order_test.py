"""
Per-family reaction order: trains a 15-param flow giving each of the 3 site families its
own order n_f, inverts the two real hydrogenated-graphene curves, and reports per-family n
posteriors, whether any family demands n~2, and posterior-predictive RMSE vs the global-n model.
Run: python -m tpdsbi.robustness.reaction_order_test    (~40 min if the flow needs training)
"""
import numpy as np
from tpdsbi.inference.neural_posterior import NPE
from tpdsbi.analysis.corpus_inversion import CORPUS, load_npe, load_curve

PRIOR_W90 = 0.9 * 1.7   # 90% width of the U(0.8, 2.5) order prior


def rmse_of(npe, S, y, lb, k=150):
    pp = npe.cfg.simulate(S[np.random.default_rng(0).choice(len(S), k)], lb)
    return float(np.sqrt(((pp.mean(0) - y) ** 2).mean()))


def main():
    npe = NPE("per_family_order", n_layers=12)
    import os
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "flow_per_family_order.pt")):
        npe.train(n_sims=240000, epochs=40, seed=0)
    glob = load_npe("hydrogenated_carbon")

    print("Per-family order (15-param) vs global order (13-param)")
    for name in ["nechaev_hslg", "nechaev_flg"]:
        cfg_name, csv, lb, _ = CORPUS[name]
        y = load_curve(csv, npe.cfg)
        S = npe.invert(y, lb, 6000)
        Sg = glob.invert(y, lb, 6000)
        print(f"  {name}:")
        any2 = False
        for f in range(3):
            q = np.percentile(S[:, 11 + f], [5, 50, 95])
            con = 1 - (q[2] - q[0]) / PRIOR_W90
            demands2 = q[0] > 1.5
            any2 |= demands2
            print(f"    family {f+1}: n = {q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}] "
                  f"(contraction {con:+.2f}){'  <- demands n~2' if demands2 else ''}")
        r_p = rmse_of(npe, S, y, lb)
        r_g = rmse_of(glob, Sg, y, lb)
        print(f"    fit RMSE: per-family {r_p:.3f} | global-n {r_g:.3f} "
              f"-> {'per-family justified' if r_p < r_g - 0.01 else 'global-n parsimonious'}")
        if not any2:
            print("    no family demands n~2")


if __name__ == "__main__":
    main()
