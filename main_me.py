from typing import Tuple
from dataclasses import dataclass
import functools
import os
import time
import pickle

import jax
import jax.numpy as jnp

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from qdax.environments import behavior_descriptor_extractor
from qdax.core.map_elites import MAPElites
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import wandb
from utils import Config, get_env


@hydra.main(version_base="1.2", config_path="configs/", config_name="me")
def main(config: Config) -> None:
    wandb.init(
        project="DCG-MAP-Elites",
        name=config.algo.name,
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = get_env(config)
    reset_fn = jax.jit(env.reset)

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.num_init_cvt_samples,
        num_centroids=config.num_centroids,
        minval=config.env.min_bd,
        maxval=config.env.max_bd,
        random_key=random_key,
    )

    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.batch_size)
    fake_batch_obs = jnp.zeros(shape=(config.batch_size, env.observation_size))
    init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs)

    param_count = sum(x[0].size for x in jax.tree_util.tree_leaves(init_params))
    print("Number of parameters in policy_network: ", param_count)

    # Define the fonction to play a step with the policy in the environment
    def play_step_fn(env_state, policy_params, random_key):
        actions = policy_network.apply(policy_params, env_state.obs)
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            desc=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
            desc_prime=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
        )

        return next_state, policy_params, random_key, transition

    # Prepare the scoring function
    bd_extraction_fn = behavior_descriptor_extractor[config.env.name]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=config.env.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    @jax.jit
    def evaluate_repertoire(random_key, repertoire):
        repertoire_empty = repertoire.fitnesses == -jnp.inf

        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            repertoire.genotypes, random_key
        )

        # Compute repertoire QD score
        qd_score = jnp.sum((1.0 - repertoire_empty) * fitnesses).astype(float)

        # Compute repertoire desc error mean
        error = jnp.linalg.norm(repertoire.descriptors - descriptors, axis=1)
        dem = (jnp.sum((1.0 - repertoire_empty) * error) / jnp.sum(1.0 - repertoire_empty)).astype(float)

        return random_key, qd_score, dem

    def get_n_offspring_added(metrics):
        return jnp.sum(metrics["is_offspring_added"], axis=-1)

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = 0

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * config.env.episode_length,
    )

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=config.algo.iso_sigma, line_sigma=config.algo.line_sigma
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=config.batch_size
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_function,
    )

    # Compute initial repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(init_params, centroids, random_key)

    log_period = 10
    num_loops = int(config.num_iterations / log_period)

    metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "qd_score_repertoire", "dem_repertoire", "ga_offspring_added", "time"], jnp.array([]))
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )

    # Main loop
    map_elites_scan_update = map_elites.scan_update
    for i in range(num_loops):
        start_time = time.time()
        (repertoire, emitter_state, random_key,), current_metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # Metrics
        random_key, qd_score_repertoire, dem_repertoire = evaluate_repertoire(random_key, repertoire)

        current_metrics["iteration"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, log_period)
        current_metrics["qd_score_repertoire"] = jnp.repeat(qd_score_repertoire, log_period)
        current_metrics["dem_repertoire"] = jnp.repeat(dem_repertoire, log_period)
        current_metrics["ga_offspring_added"] = get_n_offspring_added(current_metrics)
        del current_metrics["is_offspring_added"]
        metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

        # Log
        log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)
        log_metrics["ga_offspring_added"] = jnp.sum(current_metrics["ga_offspring_added"])
        csv_logger.log(log_metrics)
        wandb.log(log_metrics)

    # Metrics
    with open("./metrics.pickle", "wb") as metrics_file:
        pickle.dump(metrics, metrics_file)

    # Repertoire
    os.mkdir("./repertoire/")
    repertoire.save(path="./repertoire/")

    # Plot
    if env.behavior_descriptor_length == 2:
        env_steps = jnp.arange(config.num_iterations) * config.env.episode_length * config.batch_size
        fig, _ = plot_map_elites_results(env_steps=env_steps, metrics=metrics, repertoire=repertoire, min_bd=config.env.min_bd, max_bd=config.env.max_bd)
        fig.savefig("./plot.png")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()
