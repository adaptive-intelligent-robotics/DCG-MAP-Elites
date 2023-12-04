from dataclasses import dataclass
from typing import Callable, Tuple

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.qdcg_gecco_emitter import QualityDCGGeccoConfig, QualityDCGGeccoEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Params, RNGKey


@dataclass
class DCGMEGeccoConfig:
    """Configuration for DCGME Algorithm"""

    env_batch_size: int = 128
    proportion_mutation_ga: float = 0.5
    actor_batch_size: int  = 256
    lengthscale: float = 0.008
    max_bd: float = 1.0

    # PG emitter
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    num_critic_training_steps: int = 3000
    num_pg_training_steps: int = 150
    batch_size: int = 100
    replay_buffer_size: int = 1_000_000
    discount: float = 0.99
    reward_scaling: float = 1.0
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    soft_tau_update: float = 0.005
    policy_delay: int = 2


class DCGMEGeccoEmitter(MultiEmitter):
    def __init__(
        self,
        config: DCGMEGeccoConfig,
        policy_network: nn.Module,
        actor_network: nn.Module,
        scoring_actor_fn,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:
        self._config = config
        self._actor_network = actor_network
        self._env = env
        self._variation_fn = variation_fn

        ga_batch_size = int(self._config.proportion_mutation_ga * config.env_batch_size)
        qpg_batch_size = config.env_batch_size - ga_batch_size

        qdcg_config = QualityDCGGeccoConfig(
            env_batch_size=qpg_batch_size,
            actor_batch_size=config.actor_batch_size,
            lengthscale=config.lengthscale,
            max_bd=config.max_bd,
            critic_hidden_layer_size=config.critic_hidden_layer_size,
            num_critic_training_steps=config.num_critic_training_steps,
            num_pg_training_steps=config.num_pg_training_steps,
            batch_size=config.batch_size,
            replay_buffer_size=config.replay_buffer_size,
            discount=config.discount,
            reward_scaling=config.reward_scaling,
            critic_learning_rate=config.critic_learning_rate,
            actor_learning_rate=config.actor_learning_rate,
            policy_learning_rate=config.policy_learning_rate,
            noise_clip=config.noise_clip,
            policy_noise=config.policy_noise,
            soft_tau_update=config.soft_tau_update,
            policy_delay=config.policy_delay,
        )

        # define the quality emitter
        q_emitter = QualityDCGGeccoEmitter(
            config=qdcg_config,
            policy_network=policy_network,
            actor_network=actor_network,
            scoring_actor_fn=scoring_actor_fn,
            env=env
        )

        # define the GA emitter
        ga_emitter = MixingEmitter(
            mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        super().__init__(emitters=(q_emitter, ga_emitter))
