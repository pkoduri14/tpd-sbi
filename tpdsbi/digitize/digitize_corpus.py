"""
Pilot digitizer: extracts 7 curves (jung_h2o/jung_co from Jung Fig 6a m/z 18/28; nechaev_flg
from Nechaev Fig 4b; fig3a-fig3d from Nechaev Fig 3, YDA/Fe/Ni/Co). QA gate: digitized peak T
must match the published Tm within a few K. Writes data/<name>.csv + a verification grid.
"""
import os, numpy as np
from tpdsbi.paths import DATA, MODELS, FIGURES
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# Source-paper page renders (200 dpi PNGs) are not shipped (copyright). To re-run: render the
# cited page of each source PDF at 200 dpi into data/source_pages/ (or set TPDSBI_PAGE_DIR).
# The extracted CSVs are shipped in data/, so reproduce.py never needs this step.
SCRATCH = os.environ.get("TPDSBI_PAGE_DIR", os.path.join(DATA, "source_pages"))


def blackmask(im, thr=110):
    return (im[..., 0] < thr) & (im[..., 1] < thr) & (im[..., 2] < thr)


def clusters_at(dark, x, gap=7):
    col = np.nonzero(dark[:, x])[0]
    if len(col) == 0:
        return np.array([])
    splits = np.where(np.diff(col) > gap)[0]
    return np.array(sorted(c.mean() for c in np.split(col, splits + 1)))


def trace_stacked(dark, x0, x1, seed_index, jump=18):
    """Track one curve in a stack of non-crossing curves by continuity from a seeded index."""
    seeds = [c[seed_index] for x in range(x0, x0 + 26)
             if len(c := clusters_at(dark, x)) > seed_index]
    prev = float(np.median(seeds))
    xs, ys = [], []
    for x in range(x0, x1):
        cy = clusters_at(dark, x)
        cand = [c for c in cy if abs(c - prev) <= jump]
        if not cand:
            continue
        prev = min(cand, key=lambda c: abs(c - prev))
        xs.append(x); ys.append(prev)
    return np.array(xs), np.array(ys)


def trace_single(dark, x0, x1):
    """Single curve: per-column mean of all black pixels (data squares only)."""
    xs, ys = [], []
    for x in range(x0, x1):
        col = np.nonzero(dark[:, x])[0]
        if len(col) == 0:
            continue
        med = np.median(col)
        col = col[np.abs(col - med) < 30]
        if len(col):
            xs.append(x); ys.append(col.mean())
    return np.array(xs), np.array(ys)


def save(name, T, V):
    Vs = np.convolve(V, np.ones(5) / 5, mode="same")
    np.savetxt(os.path.join(DATA, f"{name}.csv"),
               np.column_stack([T, Vs]), delimiter=",", header="T_K,signal", comments="")
    return Vs


def jung_curves():
    im = np.asarray(Image.open(os.path.join(SCRATCH, "jung_p5.png"))
                    .crop((380, 100, 830, 560)).convert("RGB"), dtype=float)
    dark = blackmask(im)
    dark[:, :82] = False; dark[:, 381:] = False; dark[:44, :] = False; dark[378:, :] = False
    dark[:150, 245:] = False; dark[:100, :115] = False; dark[40:105, 115:190] = False
    dark[160:240, 348:] = False
    out = {}
    for name, idx in [("jung_h2o", 0), ("jung_co", 2)]:
        xs, ys = trace_stacked(dark, 84, 380, idx)
        T = 300 - (412 - xs) * 250 / (412 - 103) + 273.15
        V = (382 - ys) / 120.4
        out[name] = (T, save(name, T, V))
    return out


def nechaev_flg():
    # Fig 4b (H-FLG): single data curve + colored Gaussians (color-filtered out).
    # Grid-read calibration: T=300 K @ px113, 800 K @ px528; value 0.0 @ py460, 171 px/unit.
    im = np.asarray(Image.open(os.path.join(SCRATCH, "nechaev_p8.png"))
                    .crop((880, 1180, 1460, 1750)).convert("RGB"), dtype=float)
    dark = blackmask(im)
    dark[:, :90] = False; dark[:, 536:] = False; dark[:105, :] = False; dark[476:, :] = False
    for (cx, cy) in [(388, 381), (425, 208), (438, 342), (478, 321)]:   # peak labels 1-4
        dark[cy - 15:cy + 15, cx - 14:cx + 14] = False
    xs, ys = trace_single(dark, 95, 534)
    T = 300 + (xs - 119.5) * 100 / 84.0          # tick-detected: 300 K @ 119.5, 84 px/100 K
    V = (460 - ys) / 171.0
    return {"nechaev_flg": (T, save("nechaev_flg", T, V))}


FIG3 = {  # name: (crop box, interior (xl, xr, yt, yb), first & last x-label values)
    "fig3a": ((270, 160, 760, 620), (55, 450, 15, 372), 1000, 1450),
    "fig3b": ((900, 160, 1320, 620), (30, 398, 15, 372), 950, 1400),
    "fig3c": ((270, 700, 760, 1160), (55, 425, 15, 378), 1000, 1400),
    "fig3d": ((900, 700, 1340, 1160), (40, 422, 15, 378), 1000, 1450),
}


def fig3_curves():
    out = {}
    for name, (box, (xl, xr, yt, yb), t0, t1) in FIG3.items():
        im = np.asarray(Image.open(os.path.join(SCRATCH, "nechaev_p7.png"))
                        .crop(box).convert("RGB"), dtype=float)
        dark = blackmask(im, thr=130)
        # bottom frame row inside the interior columns
        rows = dark[:, xl:xr].sum(1)
        ybf = int(np.nonzero(rows > (xr - xl) * 0.6)[0].max())
        # calibrate from tick marks (label-text centroids proved unreliable):
        # first tick = t0; K-per-tick from median tick spacing (50 or 100 K grids)
        tickband = dark[ybf + 2:ybf + 9, :].sum(0)
        cols = np.nonzero(tickband >= 5)[0]
        groups = np.split(cols, np.where(np.diff(cols) > 3)[0] + 1)
        ticks = np.array([g.mean() for g in groups if len(g) >= 1])
        diffs = np.diff(ticks)
        s = np.median(diffs[diffs < np.median(diffs) * 1.5])   # base spacing, skip double-gaps
        step = 50 if len(ticks) >= 7 else 100                  # sparse grids label every 100 K
        a, b = step / s, t0 - (step / s) * ticks[0]
        interior = dark.copy()
        interior[:, :xl] = False; interior[:, xr:] = False
        interior[:yt, :] = False; interior[ybf - 4:, :] = False
        xs, ys = trace_single(interior, xl, xr)
        T = a * xs + b
        ymax, ymin = ys.max(), ys.min()
        V = (ymax - ys) / max(ymax - ymin, 1)
        out[name] = (T, save(name, T, V))
    return out


def main():
    curves = {}
    curves.update(jung_curves())
    curves.update(nechaev_flg())
    curves.update(fig3_curves())
    QA = {"jung_h2o": 403, "jung_co": 403, "nechaev_flg": 660,
          "fig3a": 1298, "fig3b": 1246, "fig3c": 1258, "fig3d": 1255}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    print(f"{'curve':13s} {'n':>4s} {'T-range':>12s} {'peak T':>7s} {'published':>9s} {'dT':>5s}")
    for ax, (name, (T, V)) in zip(axes.ravel(), curves.items()):
        pk = T[np.argmax(V)]
        print(f"{name:13s} {len(T):4d} {T.min():5.0f}-{T.max():5.0f} K {pk:7.0f} {QA[name]:9d} "
              f"{pk-QA[name]:+5.0f}")
        ax.plot(T, V, "k-", lw=1)
        ax.axvline(QA[name], color="red", ls="--", lw=1)
        ax.set_title(f"{name} (peak {pk:.0f} K vs {QA[name]})", fontsize=9)
    axes.ravel()[-1].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES, "digitize_pilot.png"), dpi=130)
    print("wrote figures/digitize_pilot.png")


if __name__ == "__main__":
    main()
