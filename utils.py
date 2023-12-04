from typing import Tuple
from dataclasses import dataclass
import pickle

import pandas as pd

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from qdax import environments_v1, environments
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

from omegaconf import OmegaConf


@dataclass
class Config:
    name: str
    seed: int
    num_iterations: int
    batch_size: int

    # Archive
    num_init_cvt_samples: int
    num_centroids: int
    policy_hidden_layer_sizes: Tuple[int, ...]

@dataclass
class ConfigReproducibility:
    env_name: str
    algo_name: str
    num_evaluations: int


def get_env(config: Config):
    if config.env.version == "v1":
        if config.env.name == "hopper_uni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length)
        elif config.env.name == "walker2d_uni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length)
        elif config.env.name == "halfcheetah_uni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length)
        elif config.env.name == "ant_uni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation)
        elif config.env.name == "ant_omni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation)
        elif config.env.name == "anttrap_omni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation)
        elif config.env.name == "humanoid_uni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        elif config.env.name == "humanoid_omni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        else:
            raise ValueError("Invalid environment name.")
    elif config.env.version == "v2":
        if config.env.name == "hopper_uni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                backend=config.env.backend)
        elif config.env.name == "walker2d_uni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                backend=config.env.backend)
        elif config.env.name == "halfcheetah_uni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                backend=config.env.backend)
        elif config.env.name == "ant_uni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        elif config.env.name == "ant_omni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        elif config.env.name == "anttrap_omni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        elif config.env.name == "humanoid_uni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        elif config.env.name == "humanoid_omni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        else:
            raise ValueError("Invalid environment name.")
    else:
        raise ValueError("Invalid Brax version.")

    return env


def get_config(run_dir):
    config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return config

def get_metrics(run_dir):
    with open(run_dir / "metrics.pickle", "rb") as metrics_file:
        metrics = pickle.load(metrics_file)
    return pd.DataFrame.from_dict(metrics)

def get_log(run_dir):
    return pd.read_csv(run_dir / "log.csv")

def get_repertoire(run_dir):
    # Get config
    config = get_config(run_dir)

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = get_env(config)

    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init fake params
    random_key, random_subkey = jax.random.split(random_key)
    fake_obs = jnp.zeros(shape=(env.observation_size,))
    fake_params = policy_network.init(random_subkey, fake_obs)

    # Load repertoire
    _, reconstruction_fn = ravel_pytree(fake_params)

    # Return repertoire
    return MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=str(run_dir) + "/repertoire/")

def get_df(results_dir):
    metrics_list = []
    for env_dir in results_dir.iterdir():
        if env_dir.is_file() or env_dir.name not in ["ant_omni", "anttrap_omni", "humanoid_omni", "walker2d_uni", "halfcheetah_uni", "ant_uni", "humanoid_uni"]:
            continue
        for algo_dir in env_dir.iterdir():
            for run_dir in algo_dir.iterdir():
                # Get config and metrics
                config = get_config(run_dir)
                metrics = get_metrics(run_dir)

                # Env
                metrics["env"] = config.env.name

                # Algo
                metrics["algo"] = config.algo.name

                # Run
                metrics["run"] = run_dir.name

                # Number of Evaluations
                if config.algo.name == "me_es":
                    metrics["num_evaluations"] = metrics["iteration"] * 1050
                elif config.algo.name == "dcg_me_gecco":
                    metrics["num_evaluations"] = metrics["iteration"] * (config.batch_size + config.algo.actor_batch_size)
                else:
                    metrics["num_evaluations"] = metrics["iteration"] * config.batch_size

                # Coverage
                metrics["coverage"] /= 100

                metrics_list.append(metrics)
    return pd.concat(metrics_list, ignore_index=True)
