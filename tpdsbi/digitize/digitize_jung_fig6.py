"""
Digitize the CO2 (m/z=44) TPD curve from Jung et al. 2009 (JPC C 113:18480) Fig. 6a
(graphene-oxide reduction, beta = 30 C/min = 0.5 K/s). Stacked non-crossing curves; the CO2
trace is dark-cluster #2 from the top after masking the inset and in-plot labels.
Calibration (crop offset (380,100) of jung_p5.png @200 dpi):
  T: 50 C @ px 99, 300 C @ px 372   |   P: 1.0e-8 @ py 250, 2.0e-8 @ py 133
"""
import os, numpy as np
from tpdsbi.paths import DATA, MODELS, FIGURES
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# Source-paper page renders (200 dpi PNGs) are not shipped (copyright). To re-run: render the
# cited page of each source PDF at 200 dpi into data/source_pages/ (or set TPDSBI_PAGE_DIR).
# The extracted CSVs are shipped in data/, so reproduce.py never needs this step.
SCRATCH = os.environ.get("TPDSBI_PAGE_DIR", os.path.join(DATA, "source_pages"))


def main():
    im = np.asarray(Image.open(os.path.join(SCRATCH, "jung_p5.png"))
                    .crop((380, 100, 830, 560)).convert("RGB"), dtype=float)
    dark = (im[..., 0] < 110) & (im[..., 1] < 110) & (im[..., 2] < 110)
    H, W = dark.shape
    # interior of the plot frame; mask inset (top-right) and in-plot text labels
    # calibration (tick-detected): x: 50C @ px 103, 300C @ px 412 (frame 76-412);
    #                              y: 0.0 @ py 382, 120.4 px per 1e-8 Torr
    dark[:, :82] = False; dark[:, 381:] = False; dark[:44, :] = False; dark[378:, :] = False
    dark[:150, 245:] = False                       # inset ramp plot
    dark[:100, :115] = False                       # "5x10^-8" y-label fragment (poisoned the seed)
    dark[40:105, 115:190] = False                  # "(a)" text
    dark[160:240, 348:] = False                    # "m/z" + "18" labels
    def cols_clusters(x):
        col = np.nonzero(dark[:, x])[0]
        if len(col) == 0:
            return np.array([])
        splits = np.where(np.diff(col) > 7)[0]      # merge near-splits (anti-aliasing)
        return np.array(sorted(c.mean() for c in np.split(col, splits + 1)))

    # seed: median of cluster[1] over the calm left edge
    seeds = [c[1] for x in range(84, 110) if len(c := cols_clusters(x)) >= 3]
    prev = float(np.median(seeds))
    T, V = [], []
    for x in range(84, 380):
        cy = cols_clusters(x)
        if len(cy) == 0:
            continue
        # positional candidate (2nd from top) + continuity guard against flank-splits
        cand = [c for c in cy if abs(c - prev) <= 18]
        if not cand:
            continue
        pick = min(cand, key=lambda c: abs(c - prev))
        prev = pick
        T.append(300 - (412 - x) * 250 / (412 - 103))
        V.append((382 - pick) / 120.4)              # 1e-8 Torr (tick-calibrated)
    T = np.array(T) + 273.15                        # -> K
    V = np.array(V)
    Vs = np.convolve(V, np.ones(5) / 5, mode="same")
    np.savetxt(os.path.join(DATA, "jung2009_go_co2.csv"),
               np.column_stack([T, Vs]), delimiter=",", header="T_K,P_1e8Torr", comments="")
    print(f"digitized {len(T)} cols, T {T.min():.0f}-{T.max():.0f} K, "
          f"peak {Vs.max():.2f}e-8 @ {T[np.argmax(Vs)]:.0f} K "
          f"(paper Tm(30C/min) ~ 402 K)")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.imshow(im.astype(np.uint8), extent=[(82-99)*250/273+50+273.15, (450-99)*250/273+50+273.15,
                                           (250-460)/117+1, (250-0)/117+1], aspect="auto")
    ax.plot(T, Vs, "r-", lw=1.5, label="digitized CO2 (m/z=44)")
    ax.set_xlabel("T (K)"); ax.set_ylabel("P (1e-8 Torr)"); ax.legend()
    ax.set_title("Digitization check - Jung 2009 Fig. 6a")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "digitize_jung_check.png"), dpi=140)
    print("wrote figures/digitize_jung_check.png")


if __name__ == "__main__":
    main()
