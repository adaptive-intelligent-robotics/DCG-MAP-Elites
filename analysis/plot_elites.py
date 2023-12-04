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
    "ant_omni": "Ant Omni",
    "anttrap_omni": "AntTrap Omni",
    "humanoid_omni": "Humanoid Omni",
    "walker2d_uni": "Walker Uni",
    "halfcheetah_uni": "HalfCheetah Uni",
    "ant_uni": "Ant Uni",
    "humanoid_uni": "Humanoid Uni",
}

ALGO_LIST = [
    "dcg_me",
    "pga_me",
    "ablation_ai",
]
ALGO_DICT = {
    "dcg_me": "DCG-MAP-Elites-AI",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
    "ablation_ai": "Ablation AI",
}

EMITTER_LIST = {
    "dcg_me": ["ga_offspring_added", "qpg_ai_offspring_added"],
    "pga_me": ["ga_offspring_added", "qpg_ai_offspring_added"],
    "qd_pg": ["ga_offspring_added", "qpg_ai_offspring_added"],
    "me": ["ga_offspring_added"],
    "me_es": ["es_offspring_added"],
}
EMITTER_DICT = {
    "ga_offspring_added": "GA",
    "qpg_offspring_added": "PG",
    "dpg_offspring_added": "DPG",
    "es_offspring_added": "ES",
    "ai_offspring_added": "AI",
    "qpg_ai_offspring_added": "PG + AI",
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
    nrows = len(EMITTER_LIST[ALGO_LIST[0]])
    fig, axes = plt.subplots(nrows=nrows, ncols=len(ENV_LIST), sharex=True, squeeze=False, figsize=(25, 7))

    # Create formatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    for col, env in enumerate(ENV_LIST):
        print(env)

        # Set title for the column
        axes[0, col].set_title(ENV_DICT[env])

        # Set the x label and formatter for the column
        axes[nrows-1, col].set_xlabel(XLABEL)
        axes[nrows-1, col].xaxis.set_major_formatter(formatter)

        # Get df for the current env
        df_plot = df[(df["env"] == env) & (df["algo"].isin(ALGO_LIST))]

        # Plot
        for i, emitter in enumerate(EMITTER_LIST[ALGO_LIST[0]]):
            ax = sns.lineplot(
                df_plot,
                x="num_evaluations",
                y=emitter,
                hue="algo",
                hue_order=ALGO_LIST,
                estimator=np.median,
                errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                legend=False,
                ax=axes[i, col],
            )

            if col == 0:
                axes[i, col].set_ylabel("Elites for {}".format(EMITTER_DICT[emitter]))
            else:
                axes[i, col].set_ylabel(None)

            # Customize axis
            customize_axis(axes[i, col])

    # Legend
    fig.legend(ax.get_lines(), [ALGO_DICT[algo] for algo in ALGO_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.04), ncols=len(ALGO_LIST), frameon=False)

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    # Save plot
    fig.savefig("/project/output/plot_elites.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("/project/output/")
    df = get_df(results_dir)

    # Sum PG and AI emitters
    df["qpg_ai_offspring_added"] = df["qpg_offspring_added"] + df["ai_offspring_added"]

    # Get cumulative sum of elites
    for emitter in EMITTER_DICT:
        df[emitter] = df.groupby(['env', 'algo', 'run'])[emitter].cumsum()

    # Filter
    df = df[df["num_evaluations"] <= 1_000_000]

    # Plot
    plot(df)
