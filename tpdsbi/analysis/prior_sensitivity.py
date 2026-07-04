"""
Prior-sensitivity and contraction analysis for the pilot corpus.
(a) contraction = 1 - width90(posterior)/width90(prior), per parameter and curve.
(b) headline metrics recomputed under alternative uniform nu-priors by truncating the
    existing posterior to the new support; effective-sample counts reported.

Run: python prior_analysis.py     (seconds; reuses trained flows + pilot corpus)
"""
import os, numpy as np
from tpdsbi.analysis.corpus_inversion import CORPUS, load_npe, load_curve, main_family, KB

SCEN = {"base": None, "phonon nu[10,16]": (10.0, 16.0), "low-nu [0,8]": (0.0, 8.0)}


def lnu_cols(cfg_name):
    return {"hydrogenated_carbon": [4, 7, 10], "graphene_oxide": [3, 6], "nanodiamond": [3, 6]}[cfg_name]


def truncate(S, cfg_name, box):
    """Apply the alternative nu-prior to the main family only (constraining every family is
    unphysical when the curve demands one high-nu family)."""
    if box is None:
        return S
    lo, hi = box
    _, ln = main_family(S, cfg_name)
    m = (ln >= lo) & (ln <= hi)
    return S[m]


def metrics(S, cfg_name):
    Ea, ln = main_family(S, cfg_name)
    ew = np.percentile(Ea, 95) - np.percentile(Ea, 5)
    lw = np.percentile(ln, 95) - np.percentile(ln, 5)
    tau = np.exp(np.clip(Ea, 0.05, None) / (KB * 298.0)) / 10.0 ** ln
    toom = np.log10(np.percentile(tau, 95) / max(np.percentile(tau, 5), 1e-30))
    return ew, lw, toom


def main():
    npes = {c: load_npe(c) for c in ["graphene_oxide", "hydrogenated_carbon", "nanodiamond"]}
    posts = {}
    for name, (cfg_name, csv, lb, refs) in CORPUS.items():
        npe = npes[cfg_name]
        posts[name] = (cfg_name, npe.invert(load_curve(csv, npe.cfg), lb, 8000), npe.cfg)

    # ---------- (a) contraction ----------
    print("(a) PRIOR->POSTERIOR CONTRACTION  (1 - post_width90 / prior_width90; 0 = prior-dominated)")
    names12 = ["w1","w2","mu1","sig1","lnu1","mu2","sig2","lnu2","mu3","sig3","lnu3","base"]
    agg = {}
    for name, (cfg_name, S, cfg) in posts.items():
        prior_w = 0.9 * (cfg.HI - cfg.LO)
        post_w = np.percentile(S, 95, 0) - np.percentile(S, 5, 0)
        con = 1 - post_w / prior_w
        # report the physics-critical params: main-family Ea and lnu
        Ea, ln = main_family(S, cfg_name)
        # find prior widths for a representative Ea/lnu param (family blocks share ranges)
        iEa = 2 if cfg_name == "hydrogenated_carbon" else 1
        ilu = lnu_cols(cfg_name)[0]
        cEa = 1 - (np.percentile(Ea, 95) - np.percentile(Ea, 5)) / prior_w[iEa]
        cLn = 1 - (np.percentile(ln, 95) - np.percentile(ln, 5)) / prior_w[ilu]
        agg[name] = (cEa, cLn)
        print(f"  {name:13s}  main-family Ea contraction {cEa:5.2f} | log-nu contraction {cLn:5.2f}"
              f" | all-param mean {con.mean():5.2f}")
    cE = [v[0] for v in agg.values()]; cL = [v[1] for v in agg.values()]
    print(f"  MEDIANS: Ea contraction {np.median(cE):.2f} | log-nu contraction {np.median(cL):.2f}")

    # ---------- (b) prior sensitivity of the headline ----------
    print("\n(b) HEADLINE METRICS UNDER ALTERNATIVE NU-PRIORS (truncation = exact for uniform)")
    print(f"  {'scenario':22s} {'median Ea width':>16s} {'median lnu width':>17s} "
          f"{'median tau OoM':>15s} {'min ESS':>8s}")
    for sc, box in SCEN.items():
        ews, lws, tos, ess = [], [], [], []
        for name, (cfg_name, S, cfg) in posts.items():
            St = truncate(S, cfg_name, box)
            ess.append(len(St))
            if len(St) < 100:
                continue
            ew, lw, toom = metrics(St, cfg_name)
            ews.append(ew); lws.append(lw); tos.append(toom)
        print(f"  {sc:22s} {np.median(ews):13.2f} eV {np.median(lws):14.1f} dec "
              f"{np.median(tos):15.1f} {min(ess):8d}")
    print("\n  verdict criteria: the headline survives if median Ea width stays >0.3 eV")
    print("  and tau spread stays >5 OoM under both informative-nu scenarios.")


if __name__ == "__main__":
    main()
