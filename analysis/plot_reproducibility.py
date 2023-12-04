import sys
sys.path.append("/project/")

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
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
    "dcg_me_actor",
    "pga_me",
    "qd_pg",
    "me",
    "me_es",
]
ALGO_DICT = {
    "dcg_me": "DCG-MAP-Elites-AI",
    "dcg_me_actor": "Descriptor-Conditioned Actor",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
}


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

        # Get df for the current env
        df_plot = df[df["env"] == env]

        # QD score
        axes[0, col].yaxis.set_major_formatter(formatter)
        ax = sns.boxplot(
            df_plot,
            x="algo",
            y="qd_score",
            order=ALGO_LIST,
            hue="algo",
            hue_order=ALGO_LIST,
            ax=axes[0, col],
        )
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)

        if col == 0:
            axes[0, col].set_ylabel("Expected\nQD score")
        else:
            axes[0, col].set_ylabel(None)

        # Customize axis
        customize_axis(axes[0, col])

        # Coverage
        ax = sns.boxplot(
            df_plot,
            x="algo",
            y="distance",
            order=ALGO_LIST,
            hue="algo",
            hue_order=ALGO_LIST,
            ax=axes[1, col],
        )
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)

        if col == 0:
            axes[1, col].set_ylabel("Expected\nDistance to Descriptor")
        else:
            axes[1, col].set_ylabel(None)

        # Customize axis
        customize_axis(axes[1, col])

        # Max Fitness
        ax = sns.boxplot(
            df_plot,
            x="algo",
            y="max_fitness",
            order=ALGO_LIST,
            hue="algo",
            hue_order=ALGO_LIST,
            ax=axes[2, col],
        )
        ax.set(xticklabels=[])
        ax.tick_params(bottom=False)

        if col == 0:
            axes[2, col].set_ylabel("Expected\nMax Fitness")
        else:
            axes[2, col].set_ylabel(None)

        # Customize axis
        customize_axis(axes[2, col])

        # Remove x-axis label 'algo'
        ax.set_xlabel("")

    # Legend
    legend_patches = [mpatches.Patch(color=sns.color_palette()[i], label=ALGO_DICT[algo]) for i, algo in enumerate(ALGO_LIST)]
    fig.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(ALGO_LIST), frameon=False)

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    # Save plot
    fig.savefig("/project/output/plot_reproducibility.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("/project/output/reproducibility/")
    df = pd.DataFrame(columns=["env", "algo", "run", "qd_score", "distance", "max_fitness"])

    # Construct reproducibility DataFrame
    reproducibility_dict = {}
    for env_dir in results_dir.iterdir():
        if env_dir.name not in ENV_LIST:
            continue
        for algo_dir in env_dir.iterdir():
            if algo_dir.name not in ALGO_LIST:
                continue
            with open(algo_dir / "repertoire_fitnesses.pickle", "rb") as repertoire_fitnesses_file:
                repertoire_fitnesses = pickle.load(repertoire_fitnesses_file)
            with open(algo_dir / "repertoire_distances.pickle", "rb") as repertoire_distances_file:
                repertoire_distances = pickle.load(repertoire_distances_file)
            entry = pd.DataFrame.from_dict({
                "env": env_dir.name,
                "algo": algo_dir.name,
                "run": jnp.arange(repertoire_fitnesses.shape[0]),
                "qd_score": jnp.nansum(jnp.nanmean(repertoire_fitnesses, axis=1), axis=-1),
                "distance": jnp.nanmean(jnp.nanmean(repertoire_distances, axis=1), axis=-1),
                "max_fitness": jnp.nanmax(jnp.nanmean(repertoire_fitnesses, axis=1), axis=-1),
            })
            df = pd.concat([df, entry], ignore_index=True)

    # Plot
    plot(df)
