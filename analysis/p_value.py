import sys
sys.path.append("/project/")

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon, ranksums, mannwhitneyu
from statsmodels.stats.multitest import multipletests

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

ALGO_LIST = [
    "dcg_me",
    "pga_me",
    "qd_pg",
    "me",
    "me_es",
    "dcg_me_gecco",
    "ablation_ai",
    "ablation_actor",
]

METRICS_LIST = [
    "qd_score",
    "coverage",
    "max_fitness",
]

P_VALUE_LIST = [
    ["qd_score", "ant_omni", "dcg_me", "pga_me"],
    ["qd_score", "ant_omni", "dcg_me", "qd_pg"],
    ["qd_score", "ant_omni", "dcg_me", "me"],
    ["qd_score", "ant_omni", "dcg_me", "me_es"],
    ["qd_score", "ant_omni", "dcg_me", "ablation_ai"],
    ["qd_score", "ant_omni", "dcg_me", "dcg_me_gecco"],
    ["qd_score", "ant_omni", "dcg_me", "ablation_actor"],

    ["qd_score", "anttrap_omni", "dcg_me", "pga_me"],
    ["qd_score", "anttrap_omni", "dcg_me", "qd_pg"],
    ["qd_score", "anttrap_omni", "dcg_me", "me"],
    ["qd_score", "anttrap_omni", "dcg_me", "me_es"],
    ["qd_score", "anttrap_omni", "dcg_me", "ablation_ai"],
    ["qd_score", "anttrap_omni", "dcg_me", "dcg_me_gecco"],
    ["qd_score", "anttrap_omni", "dcg_me", "ablation_actor"],

    ["qd_score", "humanoid_omni", "dcg_me", "pga_me"],
    ["qd_score", "humanoid_omni", "dcg_me", "qd_pg"],
    ["qd_score", "humanoid_omni", "dcg_me", "me"],
    ["qd_score", "humanoid_omni", "dcg_me", "me_es"],
    ["qd_score", "humanoid_omni", "dcg_me", "ablation_ai"],
    ["qd_score", "humanoid_omni", "dcg_me", "dcg_me_gecco"],
    ["qd_score", "humanoid_omni", "dcg_me", "ablation_actor"],

    ["qd_score", "walker2d_uni", "dcg_me", "pga_me"],
    ["qd_score", "walker2d_uni", "dcg_me", "qd_pg"],
    ["qd_score", "walker2d_uni", "dcg_me", "me"],
    ["qd_score", "walker2d_uni", "dcg_me", "me_es"],
    ["qd_score", "walker2d_uni", "dcg_me", "ablation_ai"],
    ["qd_score", "walker2d_uni", "dcg_me", "dcg_me_gecco"],
    ["qd_score", "walker2d_uni", "dcg_me", "ablation_actor"],

    ["qd_score", "halfcheetah_uni", "dcg_me", "pga_me"],
    ["qd_score", "halfcheetah_uni", "dcg_me", "qd_pg"],
    ["qd_score", "halfcheetah_uni", "dcg_me", "me"],
    ["qd_score", "halfcheetah_uni", "dcg_me", "me_es"],
    ["qd_score", "halfcheetah_uni", "dcg_me", "ablation_ai"],
    ["qd_score", "halfcheetah_uni", "dcg_me", "dcg_me_gecco"],
    ["qd_score", "halfcheetah_uni", "dcg_me", "ablation_actor"],

    ["qd_score", "ant_uni", "dcg_me", "pga_me"],
    ["qd_score", "ant_uni", "dcg_me", "qd_pg"],
    ["qd_score", "ant_uni", "dcg_me", "me"],
    ["qd_score", "ant_uni", "dcg_me", "me_es"],
    ["qd_score", "ant_uni", "dcg_me", "ablation_ai"],
    ["qd_score", "ant_uni", "dcg_me", "dcg_me_gecco"],
    ["qd_score", "ant_uni", "dcg_me", "ablation_actor"],

    ["qd_score", "humanoid_uni", "dcg_me", "pga_me"],
    ["qd_score", "humanoid_uni", "dcg_me", "qd_pg"],
    ["qd_score", "humanoid_uni", "dcg_me", "me"],
    ["qd_score", "humanoid_uni", "dcg_me", "me_es"],
    ["qd_score", "humanoid_uni", "dcg_me", "ablation_ai"],
    ["qd_score", "humanoid_uni", "dcg_me", "dcg_me_gecco"],
    ["qd_score", "humanoid_uni", "dcg_me", "ablation_actor"],

    ["coverage", "ant_omni", "dcg_me", "pga_me"],
    ["coverage", "ant_omni", "dcg_me", "qd_pg"],
    ["coverage", "ant_omni", "dcg_me", "me"],
    ["coverage", "ant_omni", "dcg_me", "me_es"],
    ["coverage", "ant_omni", "dcg_me", "ablation_ai"],
    ["coverage", "ant_omni", "dcg_me", "dcg_me_gecco"],
    ["coverage", "ant_omni", "dcg_me", "ablation_actor"],

    ["coverage", "anttrap_omni", "dcg_me", "pga_me"],
    ["coverage", "anttrap_omni", "dcg_me", "qd_pg"],
    ["coverage", "anttrap_omni", "dcg_me", "me"],
    ["coverage", "anttrap_omni", "dcg_me", "me_es"],
    ["coverage", "anttrap_omni", "dcg_me", "ablation_ai"],
    ["coverage", "anttrap_omni", "dcg_me", "dcg_me_gecco"],
    ["coverage", "anttrap_omni", "dcg_me", "ablation_actor"],

    ["coverage", "humanoid_omni", "dcg_me", "pga_me"],
    ["coverage", "humanoid_omni", "dcg_me", "qd_pg"],
    ["coverage", "humanoid_omni", "dcg_me", "me"],
    ["coverage", "humanoid_omni", "dcg_me", "me_es"],
    ["coverage", "humanoid_omni", "dcg_me", "ablation_ai"],
    ["coverage", "humanoid_omni", "dcg_me", "dcg_me_gecco"],
    ["coverage", "humanoid_omni", "dcg_me", "ablation_actor"],

    ["coverage", "walker2d_uni", "dcg_me", "pga_me"],
    ["coverage", "walker2d_uni", "dcg_me", "qd_pg"],
    ["coverage", "walker2d_uni", "dcg_me", "me"],
    ["coverage", "walker2d_uni", "dcg_me", "me_es"],
    ["coverage", "walker2d_uni", "dcg_me", "ablation_ai"],
    ["coverage", "walker2d_uni", "dcg_me", "dcg_me_gecco"],
    ["coverage", "walker2d_uni", "dcg_me", "ablation_actor"],

    ["coverage", "halfcheetah_uni", "dcg_me", "pga_me"],
    ["coverage", "halfcheetah_uni", "dcg_me", "qd_pg"],
    ["coverage", "halfcheetah_uni", "dcg_me", "me"],
    ["coverage", "halfcheetah_uni", "dcg_me", "me_es"],
    ["coverage", "halfcheetah_uni", "dcg_me", "ablation_ai"],
    ["coverage", "halfcheetah_uni", "dcg_me", "dcg_me_gecco"],
    ["coverage", "halfcheetah_uni", "dcg_me", "ablation_actor"],

    ["coverage", "ant_uni", "dcg_me", "pga_me"],
    ["coverage", "ant_uni", "dcg_me", "qd_pg"],
    ["coverage", "ant_uni", "dcg_me", "me"],
    ["coverage", "ant_uni", "dcg_me", "me_es"],
    ["coverage", "ant_uni", "dcg_me", "ablation_ai"],
    ["coverage", "ant_uni", "dcg_me", "dcg_me_gecco"],
    ["coverage", "ant_uni", "dcg_me", "ablation_actor"],

    ["coverage", "humanoid_uni", "dcg_me", "pga_me"],
    ["coverage", "humanoid_uni", "dcg_me", "qd_pg"],
    ["coverage", "humanoid_uni", "dcg_me", "me"],
    ["coverage", "humanoid_uni", "dcg_me", "me_es"],
    ["coverage", "humanoid_uni", "dcg_me", "ablation_ai"],
    ["coverage", "humanoid_uni", "dcg_me", "dcg_me_gecco"],
    ["coverage", "humanoid_uni", "dcg_me", "ablation_actor"],

    ["max_fitness", "ant_omni", "dcg_me", "pga_me"],
    ["max_fitness", "ant_omni", "dcg_me", "qd_pg"],
    ["max_fitness", "ant_omni", "dcg_me", "me"],
    ["max_fitness", "ant_omni", "dcg_me", "me_es"],
    ["max_fitness", "ant_omni", "dcg_me", "ablation_ai"],
    ["max_fitness", "ant_omni", "dcg_me", "dcg_me_gecco"],
    ["max_fitness", "ant_omni", "dcg_me", "ablation_actor"],

    ["max_fitness", "anttrap_omni", "dcg_me", "pga_me"],
    ["max_fitness", "anttrap_omni", "dcg_me", "qd_pg"],
    ["max_fitness", "anttrap_omni", "dcg_me", "me"],
    ["max_fitness", "anttrap_omni", "dcg_me", "me_es"],
    ["max_fitness", "anttrap_omni", "dcg_me", "ablation_ai"],
    ["max_fitness", "anttrap_omni", "dcg_me", "dcg_me_gecco"],
    ["max_fitness", "anttrap_omni", "dcg_me", "ablation_actor"],

    ["max_fitness", "humanoid_omni", "dcg_me", "pga_me"],
    ["max_fitness", "humanoid_omni", "dcg_me", "qd_pg"],
    ["max_fitness", "humanoid_omni", "dcg_me", "me"],
    ["max_fitness", "humanoid_omni", "dcg_me", "me_es"],
    ["max_fitness", "humanoid_omni", "dcg_me", "ablation_ai"],
    ["max_fitness", "humanoid_omni", "dcg_me", "dcg_me_gecco"],
    ["max_fitness", "humanoid_omni", "dcg_me", "ablation_actor"],

    ["max_fitness", "walker2d_uni", "dcg_me", "pga_me"],
    ["max_fitness", "walker2d_uni", "dcg_me", "qd_pg"],
    ["max_fitness", "walker2d_uni", "dcg_me", "me"],
    ["max_fitness", "walker2d_uni", "dcg_me", "me_es"],
    ["max_fitness", "walker2d_uni", "dcg_me", "ablation_ai"],
    ["max_fitness", "walker2d_uni", "dcg_me", "dcg_me_gecco"],
    ["max_fitness", "walker2d_uni", "dcg_me", "ablation_actor"],

    ["max_fitness", "halfcheetah_uni", "dcg_me", "pga_me"],
    ["max_fitness", "halfcheetah_uni", "dcg_me", "qd_pg"],
    ["max_fitness", "halfcheetah_uni", "dcg_me", "me"],
    ["max_fitness", "halfcheetah_uni", "dcg_me", "me_es"],
    ["max_fitness", "halfcheetah_uni", "dcg_me", "ablation_ai"],
    ["max_fitness", "halfcheetah_uni", "dcg_me", "dcg_me_gecco"],
    ["max_fitness", "halfcheetah_uni", "dcg_me", "ablation_actor"],

    ["max_fitness", "ant_uni", "dcg_me", "pga_me"],
    ["max_fitness", "ant_uni", "dcg_me", "qd_pg"],
    ["max_fitness", "ant_uni", "dcg_me", "me"],
    ["max_fitness", "ant_uni", "dcg_me", "me_es"],
    ["max_fitness", "ant_uni", "dcg_me", "ablation_ai"],
    ["max_fitness", "ant_uni", "dcg_me", "dcg_me_gecco"],
    ["max_fitness", "ant_uni", "dcg_me", "ablation_actor"],

    ["max_fitness", "humanoid_uni", "dcg_me", "pga_me"],
    ["max_fitness", "humanoid_uni", "dcg_me", "qd_pg"],
    ["max_fitness", "humanoid_uni", "dcg_me", "me"],
    ["max_fitness", "humanoid_uni", "dcg_me", "me_es"],
    ["max_fitness", "humanoid_uni", "dcg_me", "ablation_ai"],
    ["max_fitness", "humanoid_uni", "dcg_me", "dcg_me_gecco"],
    ["max_fitness", "humanoid_uni", "dcg_me", "ablation_actor"],
]


if __name__ == "__main__":
    # Create the DataFrame
    results_dir = Path("/project/output/")
    df = get_df(results_dir)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_000_000]

    # Keep only the last iteration
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()
    df = df.loc[idx]

    # Compute p-values
    p_value_df = pd.DataFrame(columns=["metric", "env", "algo_1", "algo_2", "p_value"])
    for metric in METRICS_LIST:
        for env in ENV_LIST:
            for algo_1 in ALGO_LIST:
                for algo_2 in ALGO_LIST:
                    stat = mannwhitneyu(
                        df[(df["env"] == env) & (df["algo"] == algo_1)][metric],
                        df[(df["env"] == env) & (df["algo"] == algo_2)][metric],
                    )
                    p_value_df.loc[len(p_value_df)] = {"metric": metric, "env": env, "algo_1": algo_1, "algo_2": algo_2, "p_value": stat.pvalue}

    # Filter p-values
    p_value_df.set_index(["metric", "env", "algo_1", "algo_2"], inplace=True)
    p_value_df = p_value_df.loc[P_VALUE_LIST]

    # Correct p-values
    p_value_df.reset_index(inplace=True)
    p_value_df["p_value_corrected"] = multipletests(p_value_df["p_value"], method="holm")[1]
    p_value_df = p_value_df.pivot(index=["env", "algo_1", "algo_2"], columns="metric", values="p_value_corrected")

    # Save p-values
    p_value_df.to_csv("/project/output/p_value.csv")
