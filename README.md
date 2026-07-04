# tpd-sbi

Calibrated Bayesian inversion of temperature-programmed desorption (TPD/TDS) spectra via simulation-based inference.

A single heating ramp does not determine desorption parameters the way point fits imply. This code inverts a TPD curve into the full posterior over activation energy, prefactor, site-energy width, and reaction order, using a conditional normalizing flow trained on a Polanyi-Wigner forward model. The posterior is calibration-checked: 90% intervals contain the truth 90% of the time on held-out simulations, including for derived quantities like retention.

Applied to 10 digitized published curves (hydrogenated graphene, graphene oxide, nanodiamond), the posteriors show that the reported "anomalously low" carbon prefactors (K0 = 9-180 s^-1) are consistent with single-site fitting artifacts of broadened spectra: fitting single-site models to simulated heterogeneous surfaces with ordinary nu = 1e13 reproduces the published low-K0 pattern, and the calibrated posteriors on the real curves place the standard 1e13 inside the credible interval for all 10.

## Install

Python >= 3.10. From the repo root:

```bash
git clone https://github.com/pkoduri14/tpd-sbi.git
cd tpd-sbi
pip install -e .
```

Pinned versions in `requirements.txt`.

## Analyze your own curve

    python analyze_my_tpd.py --demo
    python analyze_my_tpd.py mydata.csv --beta 0.5 --hold 298

Input: two-column CSV (temperature in K, signal in any units). `--beta` is your heating rate in K/s.
Output: posterior summaries, a fit-quality check, a retention forecast at the hold temperature, and a 4-panel figure in `figures/user/`.

## Reproduce

    python -m tpdsbi.validation.regression_tests             # ~3 min; ends REGRESSION TESTS: ALL PASS
    python reproduce.py                                      # core suite, ~2 hrs
    python reproduce.py --full                               # + slow validations and robustness, ~12 hrs
    python reproduce.py --retrain                            # retrain all flows first, ~2 hrs extra

Trained networks are shipped in `models/`, so nothing needs training to run.
Figures are regenerated into `figures/` (shipped empty).

## Results (10-curve corpus)

- median hidden 90% uncertainty on Ea from a single ramp: 1.23 eV
- log10(nu) is essentially unconstrained by one ramp (~13 decades); this
  survives informative nu-priors
- room-temperature retention forecasts spread ~19 orders of magnitude
- reaction order for hydrogenated graphene: n = 1.04 [0.82, 1.31]
- negative control: 30/30 single-site nu=1e13 truths recovered, 0/30
  spuriously low-nu posteriors

## Layout

    analyze_my_tpd.py      entry point: invert your own curve
    reproduce.py           entry point: rerun the analysis suite
    tpdsbi/
      forward/             Polanyi-Wigner forward models per material class
      inference/           conditional-flow neural posterior estimation
      analysis/            corpus inversion, low-prefactor study, prior sensitivity
      validation/          calibration, regression tests, baselines, ground truth
      robustness/          ablations, seed variance, sensitivity studies
      digitize/            figure-to-CSV extraction
    data/                  10 digitized curves (CSV)
    models/                trained flows (~37 MB)
    docs/                  validation record, plain-language explainer, data sources

## Data

The curves are digitized from published figures:

- Jung et al., "Reduction Kinetics of Graphene Oxide Determined by Electrical
  Transport Measurements and Temperature Programmed Desorption",
  J. Phys. Chem. C 113, 18480-18486 (2009). DOI 10.1021/jp904396j
- Nechaev et al., "Studying the thermal desorption of hydrogen in some carbon
  nanostructures and graphite", Int. J. Hydrogen Energy 45, 25030-25042
  (2020). DOI 10.1016/j.ijhydene.2020.06.242

They are digitized approximations, not the original measurements. The extraction
scripts in `tpdsbi/digitize/` require page renders of the source PDFs, which
are not shipped (copyright); the extracted CSVs are, so nothing in
reproduction needs the PDFs.

## License

MIT (see [`LICENSE`](LICENSE))
