import os
import functools
import time
import pickle
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import serialization

from qdax.core.map_elites_dcg import MAPElitesDCG
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, get_cells_indices, MapElitesRepertoire
from qdax import environments
from qdax.environments.locomotion_wrappers import HumanoidOmniDCGWrapper, HexapodOmniDCGWrapper, AntOmniDCGWrapper, AntTrapOmniDCGWrapper, Walker2dDCGWrapper, HalfcheetahDCGWrapper
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from qdax.tasks.brax_envs import reset_based_scoring_dcg_function_brax_envs as scoring_dcg_function
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.dcg_me_emitter import DCGMEConfig, DCGMEEmitter

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

    # GA emitter
    iso_sigma: float
    line_sigma: float

    # PG emitter
    proportion_mutation_ga: float
    critic_hidden_layer_size: Tuple[int, ...]
    num_critic_training_steps: int
    num_pg_training_steps: int
    transitions_batch_size: int
    replay_buffer_size: int
    discount: float
    reward_scaling: float
    critic_learning_rate: float
    greedy_learning_rate: float
    policy_learning_rate: float
    noise_clip: float
    policy_noise: float
    soft_tau_update: float
    policy_delay: int

    # DCG-MAP-Elites
    lengthscale: float
    descriptor_sigma: float

@hydra.main(config_path="configs/", config_name="dcg_me")
def main(config: Config) -> None:
    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    if config.env_name == "humanoid_omni":
        config.episode_length = 250
        config.min_bd, config.max_bd = -30., 30.
        lengthscale = config.lengthscale * (config.max_bd - config.min_bd)
        descriptor_sigma = config.descriptor_sigma * (config.max_bd - config.min_bd)

        env = environments.create(config.env_name, episode_length=config.episode_length, fixed_init_state=True)
        env = HumanoidOmniDCGWrapper(env, config.env_name)
    elif config.env_name == "hexapod_omni":
        config.episode_length = 250
        config.min_bd, config.max_bd = -2., 2.
        lengthscale = config.lengthscale * (config.max_bd - config.min_bd)
        descriptor_sigma = config.descriptor_sigma * (config.max_bd - config.min_bd)

        env = environments.create(config.env_name, episode_length=config.episode_length)
        env = HexapodOmniDCGWrapper(env, config.env_name)
    elif config.env_name == "ant_omni":
        config.episode_length = 250
        config.min_bd, config.max_bd = -30., 30.
        lengthscale = config.lengthscale * (config.max_bd - config.min_bd)
        descriptor_sigma = config.descriptor_sigma * (config.max_bd - config.min_bd)

        env = environments.create(config.env_name, episode_length=config.episode_length, fixed_init_state=True, use_contact_forces=False)
        env = AntOmniDCGWrapper(env, config.env_name)
    elif config.env_name == "anttrap_omni":
        config.episode_length = 250
        config.min_bd, config.max_bd = -30., 30.
        lengthscale = config.lengthscale * (config.max_bd - config.min_bd)
        descriptor_sigma = config.descriptor_sigma * (config.max_bd - config.min_bd)

        env = environments.create(config.env_name, episode_length=config.episode_length, fixed_init_state=True, use_contact_forces=False)
        env = AntTrapOmniDCGWrapper(env, config.env_name)
    elif config.env_name == "walker2d_uni":
        config.episode_length = 1000
        config.min_bd, config.max_bd = 0., 1.
        lengthscale = config.lengthscale * (config.max_bd - config.min_bd)
        descriptor_sigma = config.descriptor_sigma * (config.max_bd - config.min_bd)

        env = environments.create(config.env_name, episode_length=config.episode_length, fixed_init_state=True)
        env = Walker2dDCGWrapper(env, config.env_name)
    elif config.env_name == "halfcheetah_uni":
        config.episode_length = 1000
        config.min_bd, config.max_bd = 0., 1.
        lengthscale = config.lengthscale * (config.max_bd - config.min_bd)
        descriptor_sigma = config.descriptor_sigma * (config.max_bd - config.min_bd)

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

    policy_dc_layer_sizes = config.critic_hidden_layer_size + (env.action_size,)
    policy_dc_network = MLP(
        layer_sizes=policy_dc_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.env_batch_size)
    fake_batch = jnp.zeros(shape=(config.env_batch_size, env.observation_size))
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

    # Define the fonction to play a step with the descriptor-conditioned policy in the environment
    def play_step_dcg_fn(
        env_state,
        policy_dc_params,
        descriptor,
        random_key,
    ):
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_dc_network.apply(policy_dc_params, jnp.concatenate([env_state.obs, descriptor/config.max_bd]))
        
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

        return next_state, policy_dc_params, descriptor, random_key, transition

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[config.env_name]
    scoring_dcg_fn = functools.partial(
        scoring_dcg_function,
        episode_length=config.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_dcg_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Evaluation functions
    @jax.jit
    def evaluate_repertoire(random_key, repertoire):
        repertoire_empty = repertoire.fitnesses == -jnp.inf

        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            repertoire.genotypes, random_key
        )

        # Compute repertoire QD score
        qd_score = jnp.sum((1.0 - repertoire_empty) * fitnesses).astype(float)

        # Compute repertoire distance mean
        distance = jnp.linalg.norm(repertoire.descriptors - descriptors, axis=1)
        distance_mean = (jnp.sum((1.0 - repertoire_empty) * distance) / jnp.sum(1.0 - repertoire_empty)).astype(float)

        return random_key, qd_score, distance_mean

    @jax.jit
    def evaluate_policy_dc(random_key, repertoire, greedy_policy_params):
        repertoire_empty = repertoire.fitnesses == -jnp.inf

        genotypes_dcg = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), config.num_centroids, axis=0),
                                              greedy_policy_params)
        fitnesses, descriptors, extra_scores, random_key = scoring_dcg_fn(
            genotypes_dcg, repertoire.centroids, random_key
        )

        # Compute descriptor-conditioned policy QD score
        qd_score = jnp.sum((1.0 - repertoire_empty) * fitnesses).astype(float)

        # Compute descriptor-conditioned policy distance mean
        distance = jnp.linalg.norm(repertoire.centroids - descriptors, axis=1)
        distance_mean = (jnp.sum((1.0 - repertoire_empty) * distance) / jnp.sum(1.0 - repertoire_empty)).astype(float)

        return random_key, qd_score, distance_mean

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = 0

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * config.episode_length,
    )

    # Define the PG-emitter config
    dcg_emitter_config = DCGMEConfig(
        env_batch_size=config.env_batch_size,
        batch_size=config.transitions_batch_size,
        proportion_mutation_ga=config.proportion_mutation_ga,
        critic_hidden_layer_size=config.critic_hidden_layer_size,
        critic_learning_rate=config.critic_learning_rate,
        greedy_learning_rate=config.greedy_learning_rate,
        policy_learning_rate=config.policy_learning_rate,
        noise_clip=config.noise_clip,
        policy_noise=config.policy_noise,
        discount=config.discount,
        reward_scaling=config.reward_scaling,
        replay_buffer_size=config.replay_buffer_size,
        soft_tau_update=config.soft_tau_update,
        num_critic_training_steps=config.num_critic_training_steps,
        num_pg_training_steps=config.num_pg_training_steps,
        policy_delay=config.policy_delay,
        min_bd=config.min_bd,
        max_bd=config.max_bd,
        lengthscale=lengthscale
    )

    # Get the emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=config.iso_sigma, line_sigma=config.line_sigma
    )

    pg_emitter = DCGMEEmitter(
        config=dcg_emitter_config,
        policy_network=policy_network,
        policy_dc_network=policy_dc_network,
        env=env,
        variation_fn=variation_fn,
    )

    # Instantiate MAP Elites
    map_elites = MAPElitesDCG(
        scoring_function=scoring_fn,
        scoring_dcg_fn=scoring_dcg_fn,
        emitter=pg_emitter,
        metrics_function=metrics_function,
        descriptor_sigma=descriptor_sigma,
    )

    # compute initial repertoire
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    log_period = 10
    num_loops = int(config.num_iterations / log_period)

    csv_logger = CSVLogger(
        "./log.csv",
        header=["loop",
                "iteration",
                "qd_score",
                "max_fitness",
                "mean_fitness",
                "coverage",
                "qd_score_dcg",
                "qd_score_repertoire",
                "distance_mean_dcg",
                "distance_mean_repertoire",
                "mutation_ga_count",
                "mutation_pg_count",
                "time"]
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
            if key in ["mutation_ga_count", "mutation_pg_count"]:
                # sum values
                logged_metrics[key] = jnp.sum(value)
            else:
                # take last value
                logged_metrics[key] = value[-1]

            # take all values
            if key in all_metrics.keys():
                all_metrics[key] = jnp.concatenate([all_metrics[key], value])
            else:
                all_metrics[key] = value

        random_key, qd_score_repertoire, distance_mean_repertoire = evaluate_repertoire(random_key, repertoire)
        random_key, qd_score_dcg, distance_mean_dcg = evaluate_policy_dc(random_key, repertoire, emitter_state.greedy_policy_params)

        logged_metrics["qd_score_dcg"] = qd_score_dcg
        logged_metrics["qd_score_repertoire"] = qd_score_repertoire
        logged_metrics["distance_mean_dcg"] = distance_mean_dcg
        logged_metrics["distance_mean_repertoire"] = distance_mean_repertoire

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

    # Greedy policy
    state_dict = serialization.to_state_dict(emitter_state.greedy_policy_params)
    with open("./policy.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()
