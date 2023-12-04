import functools
from pathlib import Path
import time
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization

from qdax.environments import behavior_descriptor_extractor
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from qdax.tasks.brax_envs import reset_based_scoring_actor_dc_function_brax_envs as scoring_actor_dc_function

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from utils import ConfigReproducibility, get_env, get_config, get_repertoire


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
ENV_MAX_EVALUATIONS = {
    "ant_omni": 128,
    "anttrap_omni": 128,
    "humanoid_omni": 8,
    "walker2d_uni": 64,
    "halfcheetah_uni": 64,
    "ant_uni": 64,
    "humanoid_uni": 8,
}

ALGO_LIST = [
    "dcg_me",
    "pga_me",
    "qd_pg",
    "me",
    "me_es",
]
ALGO_DICT = {
    "dcg_me": "DCG-MAP-Elites",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
}


def repertoire_reproduciblity(run_dir, num_evaluations):
    # Get config
    config = get_config(run_dir)
    assert num_evaluations <= ENV_MAX_EVALUATIONS[config.env.name] or num_evaluations % ENV_MAX_EVALUATIONS[config.env.name] == 0

    # Get repertoire
    repertoire = get_repertoire(run_dir)

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = get_env(config)
    reset_fn = jax.jit(env.reset)

    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

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

    # Prepare scan scoring function
    def scan_scoring_fn(carry, x):
        (random_keys,) = x
        fitnesses, descriptors, _, _ = jax.vmap(scoring_fn, in_axes=(None, 0))(repertoire.genotypes, random_keys)
        return (), (fitnesses, jnp.linalg.norm(descriptors - repertoire.centroids, axis=-1),)

    # Evaluate the repertoire
    length = num_evaluations // ENV_MAX_EVALUATIONS[config.env.name]

    if length == 0:
        random_keys = jax.random.split(random_key, num_evaluations)
        fitnesses, descriptors, _, _ = jax.vmap(scoring_fn, in_axes=(None, 0))(repertoire.genotypes, random_keys)
        distances = jnp.linalg.norm(descriptors - repertoire.centroids, axis=-1)

        # Place nan in fitnesses and distances according to repertoire
        fitnesses = jnp.where(jnp.isneginf(repertoire.fitnesses), jnp.nan, fitnesses)
        distances = jnp.where(jnp.isneginf(repertoire.fitnesses), jnp.nan, distances)
    else:
        random_keys = jax.random.split(random_key, num=length*ENV_MAX_EVALUATIONS[config.env.name])
        random_keys = jnp.reshape(random_keys, (length, ENV_MAX_EVALUATIONS[config.env.name], -1))
        _, (fitnesses, distances,) = jax.lax.scan(
            scan_scoring_fn,
            (),
            (random_keys,),
            length=length,
        )

        # Place nan in fitnesses and distances according to repertoire
        fitnesses = jnp.where(jnp.isneginf(repertoire.fitnesses), jnp.nan, fitnesses)
        distances = jnp.where(jnp.isneginf(repertoire.fitnesses), jnp.nan, distances)

        # Reshape fitnesses and descriptors
        fitnesses = jnp.reshape(fitnesses, (num_evaluations, repertoire.centroids.shape[0]))
        distances = jnp.reshape(distances, (num_evaluations, repertoire.centroids.shape[0]))

    return fitnesses, distances

def actor_reproduciblity(run_dir, num_evaluations):
    # Get config
    config = get_config(run_dir)
    assert num_evaluations <= ENV_MAX_EVALUATIONS[config.env.name] or num_evaluations % ENV_MAX_EVALUATIONS[config.env.name] == 0

    # Get repertoire
    repertoire = get_repertoire(run_dir)

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = get_env(config)
    reset_fn = jax.jit(env.reset)

    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    actor_dc_network = MLPDC(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Get descriptor-conditioned actor
    random_key, subkey = jax.random.split(random_key)
    fake_obs = jnp.zeros(shape=(env.observation_size,))
    fake_desc = jnp.zeros(shape=(env.behavior_descriptor_length,))
    actor_dc_params = actor_dc_network.init(subkey, fake_obs, fake_desc)

    with open(run_dir / "actor.pickle", "rb") as params_file:
        state_dict = pickle.load(params_file)
    actor_dc_params = serialization.from_state_dict(actor_dc_params, state_dict)

    def normalize_desc(desc):
        return 2*(desc - env.behavior_descriptor_limits[0])/(env.behavior_descriptor_limits[1] - env.behavior_descriptor_limits[0]) - 1

    # Define the fonction to play a step with the policy in the environment
    def play_step_actor_dc_fn(env_state, actor_dc_params, desc, random_key):
        desc_prime_normalized = normalize_desc(desc)
        actions = actor_dc_network.apply(actor_dc_params, env_state.obs, desc_prime_normalized)
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

        return next_state, actor_dc_params, desc, random_key, transition

    # Prepare the scoring function
    bd_extraction_fn = behavior_descriptor_extractor[config.env.name]
    scoring_fn = jax.jit(functools.partial(
        scoring_actor_dc_function,
        episode_length=config.env.episode_length,
        play_reset_fn=reset_fn,
        play_step_actor_dc_fn=play_step_actor_dc_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    ))

    # Prepare scan scoring function
    def scan_scoring_fn(carry, x):
        (random_keys,) = x
        (actor_dc_params,) = carry
        fitnesses, descriptors, _, _ = jax.vmap(scoring_fn, in_axes=(None, None, 0))(actor_dc_params, repertoire.centroids, random_keys)
        return (actor_dc_params,), (fitnesses, jnp.linalg.norm(descriptors - repertoire.centroids, axis=-1),)

    # Evaluate the repertoire
    length = num_evaluations // ENV_MAX_EVALUATIONS[config.env.name]

    if length == 0:
        random_keys = jax.random.split(random_key, num_evaluations)
        actor_dc_params = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), config.num_centroids, axis=0), actor_dc_params)
        fitnesses, descriptors, _, _ = jax.vmap(scoring_fn, in_axes=(None, None, 0))(actor_dc_params, repertoire.centroids, random_keys)
        distances = jnp.linalg.norm(descriptors - repertoire.centroids, axis=-1)

        # Place nan in fitnesses and distances according to repertoire
        fitnesses = jnp.where(jnp.isneginf(repertoire.fitnesses), jnp.nan, fitnesses)
        distances = jnp.where(jnp.isneginf(repertoire.fitnesses), jnp.nan, distances)
    else:
        random_keys = jax.random.split(random_key, num=length*ENV_MAX_EVALUATIONS[config.env.name])
        random_keys = jnp.reshape(random_keys, (length, ENV_MAX_EVALUATIONS[config.env.name], -1))
        actor_dc_params = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), config.num_centroids, axis=0), actor_dc_params)
        _, (fitnesses, distances,) = jax.lax.scan(
            scan_scoring_fn,
            (actor_dc_params,),
            (random_keys,),
            length=length,
        )

        # Place nan in fitnesses and distances according to repertoire
        fitnesses = jnp.where(jnp.isneginf(repertoire.fitnesses), jnp.nan, fitnesses)
        distances = jnp.where(jnp.isneginf(repertoire.fitnesses), jnp.nan, distances)

        # Reshape fitnesses and descriptors
        fitnesses = jnp.reshape(fitnesses, (num_evaluations, repertoire.centroids.shape[0]))
        distances = jnp.reshape(distances, (num_evaluations, repertoire.centroids.shape[0]))

    return fitnesses, distances

def repertoire_reproduciblity_algo(algo_dir, actor, num_evaluations):
    assert algo_dir.name == "dcg_me" or not actor, "actor only for dcg_me"
    fitnesses_list = []
    distances_list = []
    for run_dir in algo_dir.iterdir():
        start_time = time.time()

        # Filter ablations
        config = get_config(run_dir)
        if (config.algo.name == "dcg_me") and ("dc_actor" not in config.algo):
            config.algo.dc_actor = True
        if config.algo.name == "dcg_me" and config.algo.dc_actor == False and config.algo.ai_batch_size == 0:
            continue
        elif config.algo.name == "dcg_me" and config.algo.actor_batch_size == 0 and config.algo.qpg_batch_size == 128 and config.algo.ai_batch_size == 0:
            continue
        elif config.algo.name == "dcg_me" and config.algo.actor_batch_size == 64 and config.algo.qpg_batch_size == 128 and config.algo.ai_batch_size == 0:
            continue
        else:
            assert algo_dir.name == config.algo.name, "algo_dir.name != config.algo.name"

        # Evaluate repertoire
        if actor:
            fitnesses, distances = actor_reproduciblity(run_dir, num_evaluations)
        else:
            fitnesses, distances = repertoire_reproduciblity(run_dir, num_evaluations)
        fitnesses_list.append(fitnesses)
        distances_list.append(distances)

        # Display information
        end_time = time.time()
        print(len(fitnesses_list), run_dir, f"time: {end_time - start_time}")
    return jnp.stack(fitnesses_list), jnp.stack(distances_list)


@hydra.main(version_base="1.2", config_path="configs/", config_name="config_reproducibility")
def main(config: ConfigReproducibility) -> None:
    # Evaluate repertoire
    results_dir = Path("/project/output/")

    # Get algo_dir
    if config.algo_name == "dcg_me_actor":
        algo_dir = results_dir / config.env_name / "dcg_me"
        actor = True
    else:
        algo_dir = results_dir / config.env_name / config.algo_name
        actor = False

    # Compute repoducbility
    fitnesses, distances = repertoire_reproduciblity_algo(algo_dir, actor, config.num_evaluations)

    # Save results
    with open("./repertoire_fitnesses.pickle", "wb") as fitnesses_file:
        pickle.dump(fitnesses, fitnesses_file)

    with open("./repertoire_distances.pickle", "wb") as distances_file:
        pickle.dump(distances, distances_file)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=ConfigReproducibility)
    main()
