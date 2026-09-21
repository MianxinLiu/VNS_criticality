"""Render manuscript Figures 2–6."""

from plot_function import (
    plot_figure2_criticality,
    plot_figure3_fei,
    plot_figure4_complexity,
    plot_figure5_behavior,
    plot_figure6_rww,
    plot_figure6_rww_bw,
    plot_figure6_rww_bw_stability,
    setup_style,
)


def main():
    setup_style()
    plot_figure2_criticality()
    plot_figure3_fei()
    plot_figure4_complexity()
    plot_figure5_behavior()
    plot_figure6_rww()
    plot_figure6_rww_bw()
    plot_figure6_rww_bw_stability()


if __name__ == "__main__":
    main()
