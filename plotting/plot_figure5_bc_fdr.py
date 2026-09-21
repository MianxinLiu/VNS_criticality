"""Figure 4B/C effect sizes with raw and within-panel BH-adjusted p values."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import plot_manuscript_figures_2_6 as source


def main():
    source.setup_style()
    tree = ast.parse(Path(source.__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plot_figure5_behavior')
    assignment = next(n for n in function.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'lmm' for x in n.targets))
    models = pd.DataFrame(ast.literal_eval(assignment.value), columns=['hemi', 'metric', 'effect', 'p_raw'])
    correlations = pd.DataFrame(source.behavior_change_correlation_rows()).rename(columns={'r': 'effect', 'p': 'p_raw'})
    frames = [correlations, models]
    for frame, panel in zip(frames, ['B', 'C']):
        assert len(frame) == 18 and not frame.duplicated(['metric', 'hemi']).any()
        frame['panel'] = panel
        frame['region'] = frame.hemi.map(dict(source.HEMI_SPECS))
        frame['p_fdr'] = multipletests(frame.p_raw, method='fdr_bh')[1]
        # Independent BH calculation to verify values and the correction family.
        order = np.argsort(frame.p_raw.to_numpy())
        expected = np.minimum.accumulate((frame.p_raw.to_numpy()[order] * 18 / np.arange(1, 19))[::-1])[::-1].clip(0, 1)
        np.testing.assert_allclose(frame.p_fdr.to_numpy()[order], expected)
    stem = Path('figures/paper_draft_0730/supplementary_figure5_bc_fdr')
    pd.concat(frames, ignore_index=True).to_csv(stem.with_suffix('.csv'), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    fig.subplots_adjust(left=.18, right=.98, top=.88, bottom=.15, wspace=.16)
    metrics = ['dis_cri', 'fEI', 'SE', 'DFA', 'MSE', 'Composite']
    hemis = ['ipsi', 'contra', 'bilateral']
    colors = ['#D96545', '#2E8B73', '#C4972F']
    for ax, frame, panel in zip(axes, frames, ['A', 'B']):
        is_corr = panel == 'A'
        p_positions = [1.18, 1.75] if is_corr else [.38, .55]
        ax.axvline(0, color='#89929B', lw=1)
        labels, positions = [], []
        for mi, metric in enumerate(metrics):
            for hi, hemi in enumerate(hemis):
                row = frame[(frame.metric == metric) & (frame.hemi == hemi)].iloc[0]
                y = (5-mi)*4 + 2-hi
                labels.append(f'{metric} / {row.region}')
                positions.append(y)
                if is_corr:
                    ax.plot([row.lo, row.hi], [y, y], color=colors[hi], lw=2)
                    ax.scatter(row.effect, y, color=colors[hi], s=35, zorder=3)
                else:
                    ax.barh(y, row.effect, height=.5, color=colors[hi], alpha=.85)
                for xpos, key in zip(p_positions, ['p_raw', 'p_fdr']):
                    value = row[key]
                    ax.text(xpos, y, f'{value:.3g}' if value < .001 else f'{value:.4f}', ha='center', va='center', fontsize=9, fontweight='bold' if value < .05 else 'normal')
            if mi < 5:
                ax.axhline((5-mi)*4-1, color='#E6E9ED', lw=.7)
        for xpos, title in zip(p_positions, ['Raw p', 'FDR-corrected p']):
            ax.text(xpos, 23.4, title, ha='center', fontsize=9)
        ax.set_xlim((-1.05, 2.1) if is_corr else (-.28, .66))
        ax.set_xticks([-1, -.5, 0, .5, 1] if is_corr else [-.2, -.1, 0, .1, .2])
        ax.set_ylim(-1, 24)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=9)
        ax.tick_params(axis='y', length=0)
        ax.set_xlabel('Pearson r with two-sided 95% CI' if is_corr else 'Standardized metric beta', loc='left')
        ax.set_title('Cohort 2 change-score correlations (n = 9)' if is_corr else 'Cohort 1 longitudinal FMA-UE mixed models', fontsize=12, pad=32)
        ax.text(-.03, 1.075, panel, transform=ax.transAxes, fontsize=16, fontweight='bold')
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
    fig.text(.18, .055, 'Benjamini–Hochberg FDR correction applied separately to the 18 tests in each panel. Bold: p < 0.05.\n'
             'A: full sample, including active and sham participants; two-sided Pearson tests; unadjusted 95% CIs.\n'
             'B: model estimates and raw p values retained at the precision stored in the original Figure 5 script.', fontsize=10, linespacing=1.6)
    for suffix in ['.png', '.pdf']:
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches='tight')
    plt.close(fig)
    stem.with_suffix('.md').write_text('''Supplementary Figure. Raw and FDR-corrected p values for manuscript Figure 5B and C.

B, cohort 2 associations between post-minus-pre EEG metric changes and FMA-UE improvement in the full sample (n=9), including active and sham participants. Points show Pearson r and horizontal lines show unadjusted, two-sided approximate 95% Fisher-z confidence intervals. Raw p values use two-sided Pearson tests; the delta FMA-UE=8 participant is retained.

C, cohort 1 longitudinal mixed-model standardized metric coefficients, reproduced from the stored estimates in the original Figure 4 plotting script. No confidence intervals are inferred for these coefficients. Raw p values and coefficients retain the precision stored in that script; consequently, adjusted p values for C inherit this rounding limitation. Models were not refitted for this figure.

Benjamini–Hochberg correction is applied separately within B and C, each comprising 18 tests (six metrics by three regions). Bold p values indicate values below 0.05 in the respective raw or adjusted column. Region labels follow the original Figure 4. The original manuscript figure is unchanged.

Reproduce: `python plotting/plot_figure5_bc_fdr.py` from the release root in the eegproc environment.
''')
    for frame in frames:
        print(frame[['panel', 'metric', 'region', 'p_raw', 'p_fdr']].to_string(index=False))


if __name__ == '__main__':
    main()
