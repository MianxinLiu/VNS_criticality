Supplementary Figure. Raw and FDR-corrected p values for Figure 4B and C.

B, cohort 2 associations between post-minus-pre EEG metric changes and FMA-UE improvement in the full sample (n=9), including active and sham participants. Points show Pearson r and horizontal lines show unadjusted, two-sided approximate 95% Fisher-z confidence intervals. Raw p values use two-sided Pearson tests; the delta FMA-UE=8 participant is retained.

C, cohort 1 longitudinal mixed-model standardized metric coefficients, reproduced from the stored estimates in the original Figure 4 plotting script. No confidence intervals are inferred for these coefficients. Raw p values and coefficients retain the precision stored in that script; consequently, adjusted p values for C inherit this rounding limitation. Models were not refitted for this figure.

Benjamini–Hochberg correction is applied separately within B and C, each comprising 18 tests (six metrics by three regions). Bold p values indicate values below 0.05 in the respective raw or adjusted column. Region labels follow the original Figure 4. The original manuscript figure is unchanged.

Reproduce: `python scripts/plot_figure4_bc_fdr.py` from the repository root in the eegproc environment.
