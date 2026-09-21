# Manuscript Figure 2–6 analysis code

## Figure mapping

- `plotting/plot_function.py`: primary renderer for manuscript Figures 2, 3, 4, 5 and 6. 
- `plotting/plot_figures_2_6.py`: release entry point that renders manuscript Figures 2–6 and does not require supplementary-figure inputs.
- `plotting/plot_figure5_bc_fdr.py`: retained manuscript Figure 5B/C sensitivity plot with raw and Benjamini–Hochberg FDR-corrected p values.

## Figure 5 model-analysis code

- `modelling/dmf_rww_ei_jax_gpu_fc_only.py`: base rWW model.
- `modelling/dmf_rww_ei_jax_gpu_bw_fc_only.py`: Balloon–Windkessel rWW fitting and simulation.
- `modelling/model_signal_fei_compare_bw_rww.py`: simulated-signal fEI/dis_cri comparison.
- `modelling/dmf_rww_bw_stability_analysis.py`: fixed-point and one-parameter stability continuation analysis.
- `modelling/summarize_bw_rww_motor_current_terms.py`: motor input-current decomposition used for Figure 5H.
- `modelling/models.py`, `bold_simu.py`, and `modelling/functionEI.py`: local model/fEI dependencies used by the Figure 5 analyses.

## Required input layout

The renderer expects to be run from this release directory. The small derived numerical results used for the figures are under `data/cohort1`, `data/cohort2`, and `data/model`; the plotting code reads these paths directly.

The plotting environment requires the project dependencies used by the original analyses (including NumPy, pandas, SciPy, statsmodels, matplotlib, MNE, JAX, and scikit-optimize). The Figure 2–6 renderer can be invoked with:

```bash
conda env create -f environment.yml
conda activate vns-figures
python plotting/plot_figures_2_6.py
```

Run `python plotting/plot_figure5_bc_fdr.py` after the primary Figure 2–6 renderer to generate the retained Figure 5B/C FDR sensitivity plot.

This release excludes raw EEG/MRI files, SC/FC matrices, and model simulation time-series/NPZ files, which are large files. The included `.npy` files are derived EEG feature summaries, not raw EEG recordings; the included Figure 5 CSV/JSON files are compact fitted or summary results.

All derived subject folders in `data/cohort1` and `data/cohort2` are anonymized as `sub01`, `sub02`, etc. Their original cohort ordering, lesion-side mapping, and FMA ordering are preserved in the plotting code.
