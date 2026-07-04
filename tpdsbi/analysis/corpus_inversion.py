"""
Invert every digitized published curve with its family's flow and report corpus-wide
metrics: per-curve Ea/log-nu 90% CIs, ridge correlation, published-value coverage,
nu=1e13 coverage, room-T tau spread; aggregate medians and self-inconsistency rate.

Run: python -m tpdsbi.analysis.corpus_inversion     (uses the shipped nets in models/)
"""
import os, numpy as np, torch
from tpdsbi.paths import DATA, MODELS, FIGURES
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tpdsbi.inference.neural_posterior import Flow, NPE

KB = 8.617333e-5
ORDER_IDX = {"hydrogenated_carbon": 11, "graphene_oxide": 7, "nanodiamond": 7}

# corpus: name -> (config, csv, log10 beta, published refs [(Ea_eV, log10K0 | None, label)])
CORPUS = {
    "jung_co2":     ("graphene_oxide", "jung2009_go_co2.csv", np.log10(0.5),
                     [(1.39, None, "their Kissinger (3-ramp)"), (1.60, None, "their transport")]),
    "jung_h2o":     ("graphene_oxide", "jung2009_go_h2o.csv", np.log10(0.5), []),
    "jung_co":      ("graphene_oxide", "jung2009_go_co.csv",  np.log10(0.5), []),
    "jung_ch4":     ("graphene_oxide", "jung2009_go_ch4.csv", np.log10(0.5), []),
    "nechaev_hslg": ("hydrogenated_carbon",   "nechaev2020_graphene_D2.csv",    np.log10(3.0),
                     [(0.25, 0.95, "their pk1 1st-ord"), (0.75, 4.70, "their pk2 1st-ord")]),
    "nechaev_flg":  ("hydrogenated_carbon",   "nechaev2020_fewlayer_D2.csv", np.log10(3.0), []),
    "fig3a":        ("nanodiamond", "nechaev2020_nanodiamond_yda.csv", np.log10(0.17),
                     [(1.005, 0.95, "their 1st-ord"), (1.97, 5.0, "their 2nd-ord")]),
    "fig3b":        ("nanodiamond", "nechaev2020_nanodiamond_fe.csv", np.log10(0.17),
                     [(1.150, 1.78, "their 1st-ord"), (2.26, 6.6, "their 2nd-ord")]),
    "fig3c":        ("nanodiamond", "nechaev2020_nanodiamond_ni.csv", np.log10(0.17),
                     [(1.213, 2.00, "their 1st-ord"), (2.38, 7.0, "their 2nd-ord")]),
    "fig3d":        ("nanodiamond", "nechaev2020_nanodiamond_co.csv", np.log10(0.17),
                     [(1.264, 2.26, "their 1st-ord"), (2.52, 7.7, "their 2nd-ord")]),
}


def load_npe(cfg_name):
    return NPE(cfg_name, n_layers=12)


def load_curve(csv, cfg):
    d = np.loadtxt(os.path.join(DATA, csv), delimiter=",", skiprows=1)
    T, V = d[:, 0], d[:, 1]
    V = V / V.max()
    base = float(V[:10].mean())
    right = float(V[-10:].mean())
    y = np.interp(cfg.T_GRID, T, V, left=base, right=right)
    # truncated mid-descent: continue the measured slope to baseline (a held plateau biases the posterior)
    if right > 3 * base:
        sl = np.polyfit(T[-25:], V[-25:], 1)[0]
        if sl < 0:
            ext = cfg.T_GRID > T[-1]
            y[ext] = np.clip(right + sl * (cfg.T_GRID[ext] - T[-1]), min(base, 0.05), None)
    return np.clip(y, 0, None)


def main_family(S, cfg_name):
    if cfg_name == "hydrogenated_carbon":
        w = np.column_stack([S[:, 0], S[:, 1], 1 - S[:, 0] - S[:, 1]])
        pick = np.argmax(w, 1)
        Ea = S[np.arange(len(S)), 2 + 3 * pick]
        ln = S[np.arange(len(S)), 4 + 3 * pick]
    elif cfg_name == "graphene_oxide":
        m = S[:, 0] >= 0.5
        Ea = np.where(m, S[:, 1], S[:, 4]); ln = np.where(m, S[:, 3], S[:, 6])
    else:  # nanodiamond
        m = S[:, 0] >= 0.5
        Ea = np.where(m, S[:, 1], S[:, 4]); ln = np.where(m, S[:, 3], S[:, 6])
    return Ea, ln


def main():
    npes = {c: load_npe(c) for c in ["graphene_oxide", "hydrogenated_carbon", "nanodiamond"]}
    rows, fits = [], {}
    for name, (cfg_name, csv, lb, refs) in CORPUS.items():
        npe = npes[cfg_name]
        y = load_curve(csv, npe.cfg)
        S = npe.invert(y, lb, 4000)
        Ea, ln = main_family(S, cfg_name)
        eq = np.percentile(Ea, [5, 50, 95]); lq = np.percentile(ln, [5, 50, 95])
        ridge = np.corrcoef(Ea, ln)[0, 1]
        tau = np.exp(np.clip(Ea, 0.05, None) / (KB * 298.0)) / 10.0 ** ln
        toom = np.log10(np.percentile(tau, 95) / max(np.percentile(tau, 5), 1e-30))
        nu13 = lq[0] <= 13.0 <= lq[2]
        # self-consistency vs published refs (marginal check on Ea, and nu when given)
        refhits = []
        for (rEa, rlnu, lab) in refs:
            hit = eq[0] <= rEa <= eq[2]
            if rlnu is not None:
                hit = hit and (lq[0] <= rlnu <= lq[2])
            refhits.append((lab, rEa, rlnu, hit))
        # posterior-predictive fit quality
        pp = npe.cfg.simulate(S[np.random.default_rng(0).choice(len(S), 150)], lb)
        rmse = float(np.sqrt(((pp.mean(0) - y) ** 2).mean()))
        nq = np.percentile(S[:, ORDER_IDX[cfg_name]], [5, 50, 95])
        rows.append(dict(name=name, cfg=cfg_name, eq=eq, lq=lq, ridge=ridge, toom=toom,
                         nu13=nu13, refhits=refhits, rmse=rmse, nq=nq))
        fits[name] = (npe.cfg.T_GRID, y, pp)

    print("=" * 100)
    print(f"{'curve':13s} {'Ea med [90% CI]':>22s} {'width':>6s} {'lognu med [CI]':>20s} "
          f"{'width':>6s} {'ridge':>6s} {'tau298 OoM':>10s} {'nu13?':>6s} {'RMSE':>6s} {'order n [CI]':>18s}")
    for r in rows:
        e, l = r["eq"], r["lq"]
        print(f"{r['name']:13s} {e[1]:5.2f} [{e[0]:5.2f},{e[2]:5.2f}] eV {e[2]-e[0]:6.2f} "
              f"{l[1]:5.1f} [{l[0]:4.1f},{l[2]:4.1f}] {l[2]-l[0]:6.1f} {r['ridge']:+6.2f} "
              f"{r['toom']:10.1f} {'in' if r['nu13'] else 'OUT':>6s} {r['rmse']:6.3f} {r['nq'][1]:4.2f} [{r['nq'][0]:4.2f},{r['nq'][2]:4.2f}]")
    print("\nself-consistency vs published values:")
    n_ref = n_hit = 0
    for r in rows:
        for (lab, rEa, rlnu, hit) in r["refhits"]:
            n_ref += 1; n_hit += hit
            print(f"   {r['name']:13s} {lab:24s} Ea={rEa:.2f}"
                  + (f", lognu={rlnu:.1f}" if rlnu is not None else "")
                  + f"  -> {'INSIDE' if hit else 'OUTSIDE'} posterior 90%")
    ew = [r["eq"][2] - r["eq"][0] for r in rows]
    lw = [r["lq"][2] - r["lq"][0] for r in rows]
    to = [r["toom"] for r in rows]
    print("\n" + "=" * 100)
    print("HEADLINE METRICS (10-curve corpus):")
    print(f"  median hidden 90% uncertainty on Ea:      {np.median(ew):.2f} eV "
          f"(range {min(ew):.2f}-{max(ew):.2f})")
    print(f"  median hidden 90% uncertainty on log nu:  {np.median(lw):.1f} decades")
    print(f"  median room-T extrapolation spread:       {np.median(to):.1f} orders of magnitude")
    print(f"  published values inside their own curve's posterior: {n_hit}/{n_ref} "
          f"-> self-inconsistency rate {100*(1-n_hit/max(n_ref,1)):.0f}%")
    print(f"  posteriors whose nu-90% includes the standard 1e13:  "
          f"{sum(r['nu13'] for r in rows)}/10")


    fig, axes = plt.subplots(2, 5, figsize=(19, 7))
    for ax, r in zip(axes.ravel(), rows):
        T, y, pp = fits[r["name"]]
        ax.plot(T, y, "k.", ms=2)
        ax.plot(T, pp[:50].T, color="C0", alpha=0.06)
        ax.set_title(r["name"], fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "pilot_fits.png"), dpi=130)
    print("\nwrote figures/pilot_fits.png")
    np.save(os.path.join(MODELS, "pilot_rows.npy"), np.array(rows, dtype=object))


if __name__ == "__main__":
    main()
