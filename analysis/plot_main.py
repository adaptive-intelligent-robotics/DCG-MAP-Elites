import sys
sys.path.append("/project/")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, PercentFormatter
import seaborn as sns

from utils import get_df


# Define env and algo names
ENV_LIST = [
    "ant_omni",
    "anttrap_omni",
    "humanoid_omni",
    "walker2d_uni",
    "halfcheetah_uni",
    "ant_uni",
    "humanoid_uni",
]
ENV_DICT = {
    "ant_omni": " Ant Omni ",
    "anttrap_omni": "AntTrap Omni",
    "humanoid_omni": "Humanoid Omni",
    "walker2d_uni": "Walker Uni",
    "halfcheetah_uni": "HalfCheetah Uni",
    "ant_uni": "   Ant Uni   ",
    "humanoid_uni": "Humanoid Uni",
}

ALGO_LIST = [
    "dcg_me",
    "dcg_me_gecco",
    "pga_me",
    "qd_pg",
    "me",
    "me_es",
]
ALGO_DICT = {
    "dcg_me": "DCG-MAP-Elites-AI",
    "dcg_me_gecco": "DCG-MAP-Elites GECCO",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
}

XLABEL = "Evaluations"


def customize_axis(ax):
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove ticks
    # ax.tick_params(axis="y", length=0)

    # Add grid
    ax.grid(which="major", axis="y", color="0.9")
    return ax


def plot(df):
    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=len(ENV_LIST), sharex=True, squeeze=False, figsize=(25, 10))

    # Create formatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    for col, env in enumerate(ENV_LIST):
        print(env)

        # Set title for the column
        axes[0, col].set_title(ENV_DICT[env])

        # Set the x label and formatter for the column
        axes[2, col].set_xlabel(XLABEL)
        axes[2, col].xaxis.set_major_formatter(formatter)

        # Get df for the current env
        df_plot = df[df["env"] == env]

        # QD score
        axes[0, col].yaxis.set_major_formatter(formatter)

        sns.lineplot(
            df_plot,
            x="num_evaluations",
            y="qd_score",
            hue="algo",
            hue_order=ALGO_LIST,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[0, col],
        )

        if col == 0:
            axes[0, col].set_ylabel("QD score")
        else:
            axes[0, col].set_ylabel(None)

        # Customize axis
        customize_axis(axes[0, col])

        # Coverage
        axes[1, col].set_ylim(0., 1.05)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))

        sns.lineplot(
            df_plot,
            x="num_evaluations",
            y="coverage",
            hue="algo",
            hue_order=ALGO_LIST,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[1, col],
        )

        if col == 0:
            axes[1, col].set_ylabel("Coverage")
        else:
            axes[1, col].set_ylabel(None)

        # Customize axis
        customize_axis(axes[1, col])

        # Max fitness
        ax = sns.lineplot(  # store ax for legend
            df_plot,
            x="num_evaluations",
            y="max_fitness",
            hue="algo",
            hue_order=ALGO_LIST,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[2, col],
        )

        if col == 0:
            axes[2, col].set_ylabel("Max Fitness")
        else:
            axes[2, col].set_ylabel(None)

        # Customize axis
        customize_axis(axes[2, col])

    # Legend
    fig.legend(ax.get_lines(), [ALGO_DICT[algo] for algo in ALGO_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(ALGO_LIST), frameon=False)

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    # Save plot
    fig.savefig("/project/output/plot_main.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("/project/output/")
    df = get_df(results_dir)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_000_000]

    # Plot
    plot(df)
