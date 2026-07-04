"""
Digitize the TDS spectrum from Nechaev et al. 2020 (IJHE 45:25030) Fig. 4a: hydrogenated
epitaxial single-layer graphene (H-SLG), D2 signal, beta = 3 K/s. Extracts the black data
squares (colored fit lines excluded), per-column median. Calibration (crop offset (200,1180)
of page 8 at 200 dpi): x: T=300 K @ px 143, 800 K @ px 597; y: 0.0 @ px 461, 4.0 @ px 115.
Output: data/nechaev2020_graphene_D2.csv + a verification overlay.
"""
import os, numpy as np
from tpdsbi.paths import DATA, MODELS, FIGURES
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# Source-paper page renders (200 dpi PNGs) are not shipped (copyright). To re-run: render the
# cited page of each source PDF at 200 dpi into data/source_pages/ (or set TPDSBI_PAGE_DIR).
# The extracted CSVs are shipped in data/, so reproduce.py never needs this step.
SCRATCH = os.environ.get("TPDSBI_PAGE_DIR", os.path.join(DATA, "source_pages"))

X300, X800 = 143.0, 597.0
Y0, Y4 = 461.0, 115.0


def main():
    im = np.asarray(Image.open(os.path.join(SCRATCH, "nechaev_p8.png"))
                    .crop((200, 1180, 900, 1750)).convert("RGB"), dtype=float)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    # black data squares: dark and unsaturated (colored fit lines have channel spread)
    black = (R < 120) & (G < 120) & (B < 120) & (np.abs(R - G) < 50) & (np.abs(G - B) < 50)
    # plot interior only (exclude spines/ticks/axis labels)
    ys, xs = np.nonzero(black)
    keep = (xs > 118) & (xs < 640) & (ys > 95) & (ys < 456)
    ys, xs = ys[keep], xs[keep]
    # mask the small in-plot text labels "1".."4" (located from the grid overlay)
    for (x0, x1, y0, y1) in [(332, 356, 396, 424), (490, 514, 283, 311),
                             (464, 490, 222, 250), (508, 534, 310, 338)]:
        m = ~((xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1))
        xs, ys = xs[m], ys[m]

    # per-column median with outlier rejection
    T, V = [], []
    for x in np.unique(xs):
        yy = ys[xs == x]
        med = np.median(yy)
        yy = yy[np.abs(yy - med) < 25]          # reject stray pixels
        if len(yy) == 0:
            continue
        T.append(300 + (x - X300) * 500 / (X800 - X300))
        V.append((Y0 - np.mean(yy)) * 4.0 / (Y0 - Y4))
    T, V = np.array(T), np.array(V)
    o = np.argsort(T); T, V = T[o], V[o]
    # light smoothing (the squares are chunky)
    Vs = np.convolve(V, np.ones(5) / 5, mode="same")

    np.savetxt(os.path.join(HERE, "nechaev2020_graphene_D2.csv"),
               np.column_stack([T, Vs]), delimiter=",", header="T_K,PD2_norm", comments="")
    print(f"digitized {len(T)} columns, T {T.min():.0f}-{T.max():.0f} K, "
          f"peak {Vs.max():.2f} @ {T[np.argmax(Vs)]:.0f} K")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.imshow(im.astype(np.uint8), extent=[300 + (0 - X300) * 500 / (X800 - X300),
                                           300 + (700 - X300) * 500 / (X800 - X300),
                                           (Y0 - 570) * 4 / (Y0 - Y4), (Y0 - 0) * 4 / (Y0 - Y4)],
              aspect="auto")
    ax.plot(T, Vs, "r-", lw=1.6, label="digitized")
    ax.set_xlim(280, 830); ax.set_ylim(-0.6, 4.4)
    ax.set_xlabel("T (K)"); ax.set_ylabel("P_D2 (norm units)")
    ax.set_title("Digitization check - Nechaev 2020 Fig. 4a (H-SLG)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "digitize_check.png"), dpi=140)
    print("wrote figures/digitize_check.png")


if __name__ == "__main__":
    main()
