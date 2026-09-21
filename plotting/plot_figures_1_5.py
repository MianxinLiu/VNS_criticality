"""Render the manuscript Figure 1–5 panels without supplementary outputs."""

from plot_paper_draft_0730_figures import (
    plot_figure1_criticality,
    plot_figure2_fei,
    plot_figure3_complexity,
    plot_figure4_behavior,
    plot_figure5_rww,
    plot_figure5_rww_bw,
    plot_figure5_rww_bw_stability,
    setup_style,
)


def main():
    setup_style()
    plot_figure1_criticality()
    plot_figure2_fei()
    plot_figure3_complexity()
    plot_figure4_behavior()
    plot_figure5_rww()
    plot_figure5_rww_bw()
    plot_figure5_rww_bw_stability()


if __name__ == "__main__":
    main()
