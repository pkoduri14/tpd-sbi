"""
Four robustness ablations: noise-model mismatch (coverage vs true noise 0.5-8%,
plus Student-t tails), training-set size sweep, site-family count, and a point-estimate
NN baseline (cf. arXiv 2508.03371) for accuracy and calibration.
Run: python ablations.py     (~1.5 h; run after the main chain frees the CPU)
"""
import os, numpy as np, torch
from tpdsbi.paths import DATA, MODELS, FIGURES
from tpdsbi.inference.neural_posterior import NPE, Flow



def coverage_at(npe, X, lb, P, n=600):
    cov = np.zeros(npe.cfg.NP_)
    for i in range(len(X)):
        s = npe.invert(X[i], lb[i], n)
        lo, hi = np.percentile(s, [5, 95], axis=0)
        cov += (P[i] >= lo) & (P[i] <= hi)
    return cov.mean() / len(X)


def ablation_noise():
    print("(1) Noise-model robustness (flows trained at 2% Gaussian)")
    rng = np.random.default_rng(11)
    for cfg_name in ["graphene_oxide", "hydrogenated_carbon", "nanodiamond"]:
        npe = NPE(cfg_name, n_layers=12)
        cfg = npe.cfg
        P = cfg.sample_prior(200, rng)
        lb = rng.uniform(*cfg.LOG_BETA_RANGE, 200)
        clean = cfg.simulate(P, lb)
        row = [cfg_name]
        for s in [0.005, 0.01, 0.02, 0.04, 0.08]:
            X = np.clip(clean + s * rng.standard_normal(clean.shape).astype(clean.dtype), 0, None)
            row.append(f"{coverage_at(npe, X, lb, P):.2f}")
        # Student-t df=3 scaled to a 2% marginal sd
        t = rng.standard_t(3, clean.shape).astype(clean.dtype)
        X = np.clip(clean + 0.02 * t / np.sqrt(3), 0, None)
        row.append(f"{coverage_at(npe, X, lb, P):.2f}")
        print("   {:12s} noise 0.5%:{} 1%:{} 2%:{} 4%:{} 8%:{} | t3-tails:{}".format(*row))


def ablation_train_size():
    print("(2) Training-set size (graphene_oxide, 12-layer, 40 epochs)")
    rng = np.random.default_rng(12)
    for n_sims in [30000, 60000, 120000, 240000]:
        tag = f"_ab{n_sims//1000}k"
        npe = NPE("graphene_oxide", suffix=tag, n_layers=12)
        if not os.path.exists(os.path.join(MODELS, f"flow_graphene_oxide{tag}.pt")):
            npe.train(n_sims=n_sims, epochs=40, seed=5)
        cfg = npe.cfg
        P = cfg.sample_prior(200, rng); lb = rng.uniform(*cfg.LOG_BETA_RANGE, 200)
        X = cfg.simulate(P, lb, noise=0.02, rng=rng)
        cov = coverage_at(npe, X, lb, P)
        # median posterior Ea width across the test set
        wid = []
        for i in range(50):
            s = npe.invert(X[i], lb[i], 800)
            wid.append(np.percentile(s[:, 1], 95) - np.percentile(s[:, 1], 5))
        print(f"   n_sims {n_sims:6d}: coverage {cov:.2f}, median mu1-width {np.median(wid):.2f} eV")


def ablation_families():
    print("(3) Site-family count for hydrogenated_carbon (posterior-predictive RMSE on real curves)")
    # A 2-family flow isn't grid-comparable, so instead invert with the 3-family flow and read the 3rd-family weight.
    from tpdsbi.analysis.corpus_inversion import CORPUS, load_npe, load_curve
    npe = load_npe("hydrogenated_carbon")
    for name in ["nechaev_hslg", "nechaev_flg"]:
        cfg_name, csv, lb, _ = CORPUS[name]
        S = npe.invert(load_curve(csv, npe.cfg), lb, 4000)
        w3 = 1 - S[:, 0] - S[:, 1]
        q = np.percentile(w3, [5, 50, 95])
        print(f"   {name}: posterior 3rd-family weight w3 = {q[1]:.2f} [{q[0]:.2f}, {q[2]:.2f}]"
              f" -> {'3 families earned' if q[0] > 0.05 else 'data compatible with 2 families'}")


def ablation_point_nn():
    print("(4) Point-estimate NN baseline (graphene_oxide; same sims as the flow)")
    import torch.nn as nn
    rng = np.random.default_rng(21); torch.manual_seed(21)
    npe = NPE("graphene_oxide", n_layers=12)
    cfg = npe.cfg
    N = 120000
    P = cfg.sample_prior(N, rng)
    lb = rng.uniform(*cfg.LOG_BETA_RANGE, N)
    X = np.column_stack([cfg.simulate(P, lb, noise=0.02, rng=rng), lb]).astype(np.float32)
    Y = ((P - cfg.LO) / (cfg.HI - cfg.LO)).astype(np.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, 256),
                        nn.ReLU(), nn.Linear(256, cfg.NP_))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    idx = np.arange(N)
    for ep in range(15):
        rng.shuffle(idx)
        for b in range(0, N, 512):
            j = idx[b:b + 512]
            loss = ((net(Xt[j]) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    # held-out test
    M = 400
    Pt = cfg.sample_prior(M, rng); lbt = rng.uniform(*cfg.LOG_BETA_RANGE, M)
    Xv = np.column_stack([cfg.simulate(Pt, lbt, noise=0.02, rng=rng), lbt]).astype(np.float32)
    with torch.no_grad():
        pred = net(torch.tensor(Xv)).numpy()
    pred_p = cfg.LO + np.clip(pred, 0, 1) * (cfg.HI - cfg.LO)
    # naive UQ: a single global residual sigma per parameter
    with torch.no_grad():
        res_tr = (net(Xt[:20000]).numpy() - Y[:20000])
    sig = res_tr.std(0) * (cfg.HI - cfg.LO)
    nn_mae = np.abs(pred_p - Pt).mean(0)
    cov_nn = ((Pt >= pred_p - 1.645 * sig) & (Pt <= pred_p + 1.645 * sig)).mean(0)
    # flow comparison on the same test set
    fl_mae = np.zeros(cfg.NP_); cov_fl = np.zeros(cfg.NP_)
    for i in range(M):
        S = npe.invert(Xv[i, :-1], lbt[i], 600)
        fl_mae += np.abs(np.median(S, 0) - Pt[i])
        lo, hi = np.percentile(S, [5, 95], axis=0)
        cov_fl += (Pt[i] >= lo) & (Pt[i] <= hi)
    fl_mae /= M; cov_fl /= M
    print(f"   param-avg MAE:   point-NN {nn_mae.mean():.3f} | flow-median {fl_mae.mean():.3f}")
    print(f"   90% coverage:    point-NN naive-sigma {cov_nn.mean():.2f} "
          f"(range {cov_nn.min():.2f}-{cov_nn.max():.2f}) | flow {cov_fl.mean():.2f}")


if __name__ == "__main__":
    ablation_noise()
    ablation_train_size()
    ablation_families()
    ablation_point_nn()
