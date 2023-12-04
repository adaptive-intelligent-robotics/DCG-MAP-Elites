import sys
sys.path.append("/project/")

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import jax.numpy as jnp
from qdax.utils.plotting import get_voronoi_finite_polygons_2d

from utils import get_config, get_env, get_repertoire, get_df


# Define env and algo names
ENV = "humanoid_uni"

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
DESC_DICT = {
    "ant_omni": ["x Position", "y Position"],
    "anttrap_omni": ["x Position", "y Position"],
    "humanoid_omni": ["x Position", "y Position"],
    "walker2d_uni": ["Foot Contact - Leg 1", "Foot Contact - Leg 2"],
    "halfcheetah_uni": ["Foot Contact - Leg 1", "Foot Contact - Leg 2"],
    "ant_uni": ["Foot Contact - Leg 1", "Foot Contact - Leg 2", "Foot Contact - Leg 3", "Foot Contact - Leg 4"],
    "humanoid_uni": ["Foot Contact - Leg 1", "Foot Contact - Leg 2"],
}

ALGO_LIST = [
    "dcg_me",
    "pga_me",
    "qd_pg",
    "me",
    "me_es",
]
ALGO_DICT = {
    "dcg_me": "DCG-MAP-Elites-AI",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
}


def plot_2d_repertoire(ax, repertoire, minval, maxval, vmin, vmax, display_descriptors=False, cbar=False):
    """Plot a 2d map elites repertoire on the given axis."""
    assert repertoire.centroids.shape[-1] == 2, "Descriptor space must be 2d"

    repertoire_empty = repertoire.fitnesses == -jnp.inf

    # Set axes limits
    ax.set_xlim(minval[0], maxval[0])
    ax.set_ylim(minval[1], maxval[1])
    ax.set(adjustable="box", aspect="equal")

    # Create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(repertoire.centroids)

    # Colors
    cmap = matplotlib.cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    # Fill the plot with the colors
    for idx, fitness in enumerate(repertoire.fitnesses):
        if fitness > -jnp.inf:
            region = regions[idx]
            polygon = vertices[region]
            ax.fill(*zip(*polygon), alpha=0.8, color=cmap(norm(fitness)))

    # if descriptors are specified, add points location
    if display_descriptors:
        descriptors = repertoire.descriptors[~repertoire_empty]
        ax.scatter(
            descriptors[:, 0],
            descriptors[:, 1],
            c=repertoire.fitnesses[~repertoire_empty],
            cmap=cmap,
            s=10,
            zorder=0,
        )

    # Aesthetic
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cbar.ax.tick_params()
    ax.set_aspect("equal")

    return ax


def plot(repertoires, configs):
    # Check that all configs have the same env
    envs = [config.env.name for config in configs]
    assert envs.count(envs[0]) == len(envs), "All configs must have the same env"

    # Get env to get descriptor space limits
    env = get_env(configs[0])

    # Get max and min fitness for each repertoire
    min_list = [get_min_fitnesses(repertoire) for repertoire in repertoires]
    max_list = [get_max_fitnesses(repertoire) for repertoire in repertoires]

    # Get vmin and vmax
    vmin = min(min_list)
    vmax = max(max_list)

    # Get minval and maxval
    minval, maxval = env.behavior_descriptor_limits

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(repertoires), sharey=True, squeeze=False, figsize=(25, 5))

    # Plot each repertoire
    for col, (repertoire, config) in enumerate(zip(repertoires, configs)):
        # Set title for the column
        axes[0, col].set_title(ALGO_DICT[config.algo.name])

        # Set the x and y labels
        axes[0, col].set_xlabel(DESC_DICT[envs[0]][0])
        if col == 0:
            axes[0, col].set_ylabel(DESC_DICT[envs[0]][1])
        else:
            axes[0, col].set_ylabel(None)

        # Plot repertoire
        plot_2d_repertoire(axes[0, col], repertoire, minval, maxval, vmin, vmax)

    # cax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
    cax = fig.add_axes([.91, .1, .01, .754])
    cmap = matplotlib.cm.viridis
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)

    # Save plot
    fig.savefig(f"/project/output/plot_archive_{ENV}.pdf", bbox_inches="tight")
    plt.close()


def get_min_fitnesses(repertoire):
    return repertoire.fitnesses.min(initial=jnp.inf, where=(repertoire.fitnesses != -jnp.inf))


def get_max_fitnesses(repertoire):
    return repertoire.fitnesses.max(initial=-jnp.inf, where=(repertoire.fitnesses != -jnp.inf))


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=12)

    # Create the DataFrame
    results_dir = Path("/project/output/")
    df = get_df(results_dir)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_000_000]

    # Get the median QD score for each (env, algo)
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()
    df_last_iteration = df.loc[idx]
    qd_score_median = df_last_iteration.groupby(["env", "algo"])["qd_score"].median()
    df_last_iteration = df_last_iteration.join(qd_score_median, on=["env", "algo"], rsuffix="_median")

    # Get the difference between the QD score and the median for each run
    df_last_iteration["qd_score_diff_to_median"] = abs(df_last_iteration["qd_score"] - df_last_iteration["qd_score_median"])

    # Get the most representative run for each (env, algo)
    idx = df_last_iteration.groupby(['env', 'algo'])['qd_score_diff_to_median'].idxmin()
    runs = df_last_iteration.loc[idx][["env", "algo", "run"]]

    # Get run paths
    run_paths = [results_dir / ENV / algo / runs[(runs["env"] == ENV) & (runs["algo"] == algo)]["run"].item() for algo in ALGO_LIST]

    # Get configs
    configs = [get_config(run_path) for run_path in run_paths]

    # Get repertoires
    repertoires = [get_repertoire(run_path) for run_path in run_paths]

    # Plot
    plot(repertoires, configs)
