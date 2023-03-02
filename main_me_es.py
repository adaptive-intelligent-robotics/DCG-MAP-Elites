import os
import functools
import time
import pickle
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from qdax import environments
from qdax.environments.locomotion_wrappers import HumanoidOmniDCGWrapper, HexapodOmniDCGWrapper, AntOmniDCGWrapper, AntTrapOmniDCGWrapper, Walker2dDCGWrapper, HalfcheetahDCGWrapper
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.utils.sampling import sampling
from qdax.core.emitters.mees_emitter import MEESConfig, MEESEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter

from qdax.utils.plotting import plot_map_elites_results
from qdax.utils.metrics import CSVLogger, default_qd_metrics

import hydra
from hydra.core.config_store import ConfigStore


@dataclass
class Config:
    # QD
    algo_name: str
    seed: int
    num_iterations: int

    # Environment
    env_name: str
    episode_length: int
    env_batch_size: int

    # Archive
    num_init_cvt_samples: int
    num_centroids: int
    min_bd: float
    max_bd: float
    policy_hidden_layer_sizes: Tuple[int, ...]

    # ES emitter
    sample_number: int
    sample_sigma: float
    num_optimizer_steps: int
    learning_rate: float
    l2_coefficient: float
    novelty_nearest_neighbors: int
    last_updated_size: int
    exploit_num_cell_sample: int
    explore_num_cell_sample: int
    adam_optimizer: bool
    sample_mirror: bool
    sample_rank_norm: bool
    use_explore: bool

@hydra.main(config_path="configs/", config_name="me_es")
def main(config: Config) -> None:
    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    if config.env_name == "humanoid_omni":
        config.episode_length = 250
        config.min_bd, config.max_bd = -30., 30.

        env = environments.create(config.env_name, episode_length=config.episode_length, fixed_init_state=True)
        env = HumanoidOmniDCGWrapper(env, config.env_name)
    elif config.env_name == "hexapod_omni":
        config.episode_length = 250
        config.min_bd, config.max_bd = -2., 2.

        env = environments.create(config.env_name, episode_length=config.episode_length)
        env = HexapodOmniDCGWrapper(env, config.env_name)
    elif config.env_name == "ant_omni":
        config.episode_length = 250
        config.min_bd, config.max_bd = -30., 30.

        env = environments.create(config.env_name, episode_length=config.episode_length, fixed_init_state=True, use_contact_forces=False)
        env = AntOmniDCGWrapper(env, config.env_name)
    elif config.env_name == "anttrap_omni":
        config.episode_length = 250
        config.min_bd, config.max_bd = -30., 30.

        env = environments.create(config.env_name, episode_length=config.episode_length, fixed_init_state=True, use_contact_forces=False)
        env = AntTrapOmniDCGWrapper(env, config.env_name)
    elif config.env_name == "walker2d_uni":
        config.episode_length = 1000
        config.min_bd, config.max_bd = 0., 1.

        env = environments.create(config.env_name, episode_length=config.episode_length, fixed_init_state=True)
        env = Walker2dDCGWrapper(env, config.env_name)
    elif config.env_name == "halfcheetah_uni":
        config.episode_length = 1000
        config.min_bd, config.max_bd = 0., 1.

        env = environments.create(config.env_name, episode_length=config.episode_length, fixed_init_state=True)
        env = HalfcheetahDCGWrapper(env, config.env_name)
    else:
        raise ValueError("Invalid environment name.")
    reset_fn = jax.jit(env.reset)

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.num_init_cvt_samples,
        num_centroids=config.num_centroids,
        minval=config.min_bd,
        maxval=config.max_bd,
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
    fake_batch = jnp.zeros(shape=(1, env.observation_size))
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=1, axis=0)
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Define the fonction to play a step with the policy in the environment
    def play_step_fn(
        env_state,
        policy_params,
        random_key,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_network.apply(policy_params, env_state.obs)
        
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            desc=jnp.zeros(state_desc.shape) * jnp.nan,
            input_desc=jnp.zeros(state_desc.shape) * jnp.nan,
        )

        return next_state, policy_params, random_key, transition

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[config.env_name]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=config.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Prepare the scoring functions for the offspring generated folllowing
    # the approximated gradient (each of them is evaluated 30 times)
    sampling_fn = functools.partial(
        sampling,
        scoring_fn=scoring_fn,
        num_samples=30,
    )

    # Evaluation functions
    def evaluate_repertoire(random_key, repertoire):
        _scoring_fn = functools.partial(
            scoring_function,
            episode_length=config.episode_length,
            play_reset_fn=reset_fn,
            play_step_fn=play_step_fn,
            behavior_descriptor_extractor=bd_extraction_fn,
        )
        fitnesses, descriptors, extra_scores, random_key = _scoring_fn(
            repertoire.genotypes, random_key
        )

        repertoire_empty = repertoire.fitnesses == -jnp.inf
        distance = jnp.linalg.norm(repertoire.descriptors - descriptors, axis=1)
        distance_mean = jnp.sum((1.0 - repertoire_empty) * distance) / jnp.sum(1.0 - repertoire_empty)
        return random_key, float(distance_mean)

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = 0

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * config.episode_length,
    )

    # Define the MEES-emitter config
    mees_emitter_config = MEESConfig(
        sample_number=config.sample_number,
        sample_sigma=config.sample_sigma,
        sample_mirror=config.sample_mirror,
        sample_rank_norm=config.sample_rank_norm,
        num_optimizer_steps=config.num_optimizer_steps,
        adam_optimizer=config.adam_optimizer,
        learning_rate=config.learning_rate,
        l2_coefficient=config.l2_coefficient,
        novelty_nearest_neighbors=config.novelty_nearest_neighbors,
        last_updated_size=config.last_updated_size,
        exploit_num_cell_sample=config.exploit_num_cell_sample,
        explore_num_cell_sample=config.explore_num_cell_sample,
        use_explore=config.use_explore,
    )

    # Get the emitter
    mees_emitter = MEESEmitter(
        config=mees_emitter_config,
        total_generations=config.num_iterations,
        scoring_fn=scoring_fn,
        num_descriptors=env.behavior_descriptor_length,
    )

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=sampling_fn,
        emitter=mees_emitter,
        metrics_function=metrics_function,
    )

    # compute initial repertoire
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    log_period = 10
    num_loops = int(config.num_iterations / log_period)

    csv_logger = CSVLogger(
        "./log.csv",
        header=["loop", "iteration", "qd_score", "max_fitness", "mean_fitness", "coverage", "distance_mean_repertoire", "time"]
    )
    all_metrics = {}

    scan_update = map_elites.scan_update
    # main loop
    for i in range(num_loops):
        start_time = time.time()
        # main iterations
        (repertoire, emitter_state, random_key,), metrics = jax.lax.scan(
            scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # log metrics
        logged_metrics = {"time": timelapse, "loop": 1+i, "iteration": (i+1)*log_period}
        for key, value in metrics.items():
            # take last value
            logged_metrics[key] = value[-1]

            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value

        random_key, distance_mean_repertoire = evaluate_repertoire(random_key, repertoire)

        logged_metrics["distance_mean_repertoire"] = distance_mean_repertoire

        del logged_metrics["mutation_ga_count"]
        del logged_metrics["mutation_pg_count"]
        csv_logger.log(logged_metrics)

    # Plot
    env_steps = jnp.arange(config.num_iterations) * config.episode_length * config.env_batch_size
    fig, axes = plot_map_elites_results(env_steps=env_steps, metrics=all_metrics, repertoire=repertoire, min_bd=config.min_bd, max_bd=config.max_bd)
    fig.savefig("./plot.png")

    # Metrics
    with open("./metrics.pickle", "wb") as metrics_file:
        pickle.dump(all_metrics, metrics_file)

    # Repertoire
    os.mkdir("./repertoire/")
    repertoire.save(path="./repertoire/")

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()
