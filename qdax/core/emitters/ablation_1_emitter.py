"""Implements the DCG-MAP-Elites algorithm in jax for brax environments."""
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import optax
from jax import numpy as jnp

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.losses.td3_dcg_loss import make_td3_dcg_loss_fn
from qdax.core.neuroevolution.networks.networks import QModule
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey


def similarity(descriptors_1, descriptors_2, lengthscale):
    """Compute the similarity between two batches of descriptors.

    Args:
        descriptors_1: batch of descriptors, representing the observed descriptors of
            the trajectories.
        descriptors_1: batch of descriptors, representing the sampled descriptors.
        lengthscale: lengthscale of the exponential function.

    Returns:
        batch of similarity measures.
    """
    return jnp.exp(-jnp.linalg.norm(descriptors_1 - descriptors_2, axis=1)/lengthscale)


@dataclass
class DCGMEConfig:
    """Configuration for DCG-MAP-Elites"""

    env_batch_size: int = 256
    proportion_mutation_ga: float = 0.5
    num_critic_training_steps: int = 300
    num_pg_training_steps: int = 100
    min_bd: float = 30.0
    max_bd: float = -30.0

    # TD3 params
    replay_buffer_size: int = 1_000_000
    critic_hidden_layer_size: Tuple[int, ...] = (256, 256)
    critic_learning_rate: float = 3e-4
    greedy_learning_rate: float = 3e-4
    policy_learning_rate: float = 1e-3
    noise_clip: float = 0.5
    policy_noise: float = 0.2
    discount: float = 0.99
    reward_scaling: float = 1.0
    batch_size: int = 256
    soft_tau_update: float = 0.005
    policy_delay: int = 2

    lengthscale: float = 5


class DCGMEEmitterState(EmitterState):
    """Contains training state for the learner."""

    critic_params: Params
    critic_optimizer_state: optax.OptState
    greedy_policy_params: Params
    greedy_policy_opt_state: optax.OptState
    target_critic_params: Params
    target_greedy_policy_params: Params
    replay_buffer: ReplayBuffer
    random_key: RNGKey
    steps: jnp.ndarray


class Ablation1Emitter(Emitter):
    """
    A descriptor-conditioned policy gradient emitter used to implement the
    Descriptor-Conditioned MAP-Elites (DCG-MAP-Elites) algorithm.
    """

    def __init__(
        self,
        config: DCGMEConfig,
        policy_network: nn.Module,
        policy_dc_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:
        self._config = config
        self._env = env
        self._variation_fn = variation_fn
        self._policy_network = policy_network
        self._policy_dc_network = policy_dc_network

        # Init Critics
        critic_network = QModule(
            n_critics=2, hidden_layer_sizes=self._config.critic_hidden_layer_size
        )
        self._critic_network = critic_network

        # Set up the losses and optimizers - return the opt states
        self._policy_loss_fn, self._policy_dc_loss_fn, self._critic_dc_loss_fn = make_td3_dcg_loss_fn(
            policy_fn=policy_network.apply,
            policy_dc_fn=policy_dc_network.apply,
            critic_dc_fn=critic_network.apply,
            reward_scaling=self._config.reward_scaling,
            discount=self._config.discount,
            noise_clip=self._config.noise_clip,
            policy_noise=self._config.policy_noise,
        )

        # Init optimizers
        self._greedy_policy_optimizer = optax.adam(
            learning_rate=self._config.greedy_learning_rate
        )
        self._critic_optimizer = optax.adam(
            learning_rate=self._config.critic_learning_rate
        )
        self._controllers_optimizer = optax.adam(
            learning_rate=self._config.policy_learning_rate
        )

    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[DCGMEEmitterState, RNGKey]:
        """Initializes the emitter state.
        Args:
            init_genotypes: The initial population.
            random_key: A random key.
        Returns:
            The initial state of the DCGMEEmitter, a new random key.
        """

        observation_size = self._env.observation_size
        action_size = self._env.action_size
        descriptor_size = self._env.behavior_descriptor_length

        # Initialise critic, greedy and population
        random_key, subkey = jax.random.split(random_key)
        fake_obs = jnp.zeros(shape=(observation_size,))
        fake_obs_with_desc = jnp.zeros(
            shape=(observation_size+descriptor_size,))
        fake_action = jnp.zeros(shape=(action_size,))
        critic_params = self._critic_network.init(
            subkey, obs=fake_obs_with_desc, actions=fake_action
        )
        target_critic_params = jax.tree_util.tree_map(
            lambda x: x, critic_params)

        random_key, subkey = jax.random.split(random_key)
        policy_params = self._policy_network.init(
            subkey, fake_obs
        )

        random_key, subkey = jax.random.split(random_key)
        greedy_policy_params = self._policy_dc_network.init(
            subkey, fake_obs_with_desc
        )
        target_greedy_policy_params = jax.tree_util.tree_map(
            lambda x: x, greedy_policy_params)

        # Prepare init optimizer states
        critic_optimizer_state = self._critic_optimizer.init(critic_params)
        greedy_optimizer_state = self._greedy_policy_optimizer.init(
            greedy_policy_params
        )

        # Initialize replay buffer
        dummy_transition = QDTransition.init_dummy(
            observation_dim=observation_size,
            action_dim=action_size,
            descriptor_dim=descriptor_size,
        )

        replay_buffer = ReplayBuffer.init(
            buffer_size=self._config.replay_buffer_size, transition=dummy_transition
        )

        # Initial training state
        random_key, subkey = jax.random.split(random_key)
        emitter_state = DCGMEEmitterState(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            greedy_policy_params=greedy_policy_params,
            greedy_policy_opt_state=greedy_optimizer_state,
            target_critic_params=target_critic_params,
            target_greedy_policy_params=target_greedy_policy_params,
            random_key=subkey,
            steps=jnp.array(0),
            replay_buffer=replay_buffer,
        )

        return emitter_state, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: DCGMEEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Do a single DCG-MAP-Elites iteration: train descriptor-conditioned
        critics and actor, make mutations (evo and pg), score solution, fill
        replay buffer and insert back in the MAP-Elites grid.
        Args:
            repertoire: the current repertoire of genotypes
            emitter_state: the state of the emitter used
            random_key: a random key
        Returns:
            A batch of offspring, the new emitter state and a new key.
        """

        batch_size = self._config.env_batch_size

        # Mutation evo
        mutation_ga_batch_size = int(
            self._config.proportion_mutation_ga * batch_size)
        x1, _, random_key = repertoire.sample(
            random_key, mutation_ga_batch_size)
        x2, _, random_key = repertoire.sample(
            random_key, mutation_ga_batch_size)
        x_mutation_ga, random_key = self._variation_fn(x1, x2, random_key)

        # Mutation PG
        mutation_pg_batch_size = int(batch_size - mutation_ga_batch_size)
        x1, descriptors, random_key = repertoire.sample(
            random_key, mutation_pg_batch_size)
        mutation_fn = partial(
            self._mutation_function_pg,
            emitter_state=emitter_state,
        )
        x_mutation_pg = jax.vmap(mutation_fn)(x1, descriptors)

        # gather offspring
        genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            x_mutation_ga,
            x_mutation_pg
        )

        return genotypes, mutation_ga_batch_size, random_key

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: DCGMEEmitterState,
        repertoire: Optional[Repertoire],
        genotypes: Optional[Genotype],
        fitnesses: Optional[Fitness],
        descriptors: Optional[Descriptor],
        extra_scores: ExtraScores,
    ) -> DCGMEEmitterState:
        """This function gives an opportunity to update the emitter state
        after the genotypes have been scored.
        Here it is used to fill the Replay Buffer with the transitions
        from the scoring of the genotypes, and then the training of the
        critic/greedy happens. Hence the params of critic/greedy are updated,
        as well as their optimizer states.
        Args:
            emitter_state: current emitter state.
            repertoire: the current genotypes repertoire
            genotypes: unused here - but compulsory in the signature.
            fitnesses: unused here - but compulsory in the signature.
            descriptors: unused here - but compulsory in the signature.
            extra_scores: extra information coming from the scoring function,
                this contains the transitions added to the replay buffer.
        Returns:
            New emitter state where the replay buffer has been filled with
            the new experienced transitions.
        """
        # passive
        assert "transitions" in extra_scores.keys(), "Missing transitions or wrong key"
        transitions = extra_scores["transitions"]

        descriptors_repeat = jnp.repeat(
            descriptors[:, jnp.newaxis, :], transitions.obs.shape[1], axis=1)/self._config.max_bd
        transitions_with_descriptors = transitions.replace(desc=descriptors_repeat,
                                                           input_desc=descriptors_repeat)

        replay_buffer = emitter_state.replay_buffer.insert(
            transitions_with_descriptors)

        replay_buffer = replay_buffer.insert(transitions_with_descriptors)

        # replace emitter state
        emitter_state = emitter_state.replace(replay_buffer=replay_buffer)

        def scan_train_critics(
            carry: DCGMEEmitterState, unused: Any
        ) -> Tuple[DCGMEEmitterState, Any]:
            emitter_state = carry
            new_emitter_state = self._train_critics(emitter_state)
            return new_emitter_state, ()

        # Train critics and greedy
        emitter_state, _ = jax.lax.scan(
            scan_train_critics,
            emitter_state,
            (),
            length=self._config.num_critic_training_steps,
        )

        return emitter_state  # type: ignore

    @partial(jax.jit, static_argnames=("self",))
    def _train_critics(self, emitter_state: DCGMEEmitterState) -> DCGMEEmitterState:
        """Apply one gradient step to critics and to the greedy policy
        (contained in carry in training_state), then soft update target critics
        and target greedy policy.
        Those updates are very similar to those made in TD3.
        Args:
            emitter_state: actual emitter state
        Returns:
            New emitter state where the critic and the greedy policy have been
            updated. Optimizer states have also been updated in the process.
        """

        # Sample a batch of transitions in the buffer
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        samples, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        # Concatenate states and descriptors and scale rewards
        samples = samples.replace(obs=jnp.concatenate([samples.obs, samples.input_desc], axis=1),
                                  next_obs=jnp.concatenate(
                                      [samples.next_obs, samples.input_desc], axis=1),
                                  rewards=similarity(samples.desc, samples.input_desc, self._config.lengthscale)*samples.rewards)

        # Update Critic
        random_key, subkey = jax.random.split(random_key)
        critic_loss, critic_gradient = jax.value_and_grad(self._critic_dc_loss_fn)(
            emitter_state.critic_params,
            emitter_state.target_greedy_policy_params,
            emitter_state.target_critic_params,
            samples,
            subkey,
        )
        critic_updates, critic_optimizer_state = self._critic_optimizer.update(
            critic_gradient, emitter_state.critic_optimizer_state
        )
        critic_params = optax.apply_updates(
            emitter_state.critic_params, critic_updates)
        # Soft update of target critic network
        target_critic_params = jax.tree_util.tree_map(
            lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
            + self._config.soft_tau_update * x2,
            emitter_state.target_critic_params,
            critic_params,
        )

        def update_policy_step(emitter_state: DCGMEEmitterState) -> DCGMEEmitterState:

            # Update greedy policy
            policy_loss, policy_gradient = jax.value_and_grad(self._policy_dc_loss_fn)(
                emitter_state.greedy_policy_params,
                emitter_state.critic_params,
                samples,
            )
            (
                policy_updates,
                policy_optimizer_state,
            ) = self._greedy_policy_optimizer.update(
                policy_gradient, emitter_state.greedy_policy_opt_state
            )
            greedy_policy_params = optax.apply_updates(
                emitter_state.greedy_policy_params, policy_updates
            )
            # Soft update of target greedy policy
            target_greedy_policy_params = jax.tree_map(
                lambda x1, x2: (1.0 - self._config.soft_tau_update) * x1
                + self._config.soft_tau_update * x2,
                emitter_state.target_greedy_policy_params,
                greedy_policy_params,
            )

            emitter_state = emitter_state.replace(
                greedy_policy_params=greedy_policy_params,
                greedy_policy_opt_state=policy_optimizer_state,
                target_greedy_policy_params=target_greedy_policy_params,
            )

            return emitter_state  # type: ignore

        # Delayed policy update - just use the emitter state
        emitter_state = jax.lax.cond(
            emitter_state.steps % self._config.policy_delay == 0,
            update_policy_step,
            lambda e_state: e_state,
            operand=emitter_state,
        )

        # Create new training state
        new_state = DCGMEEmitterState(
            critic_params=critic_params,
            critic_optimizer_state=critic_optimizer_state,
            greedy_policy_params=emitter_state.greedy_policy_params,
            greedy_policy_opt_state=emitter_state.greedy_policy_opt_state,
            target_critic_params=target_critic_params,
            target_greedy_policy_params=emitter_state.target_greedy_policy_params,
            random_key=random_key,
            steps=emitter_state.steps + 1,
            replay_buffer=replay_buffer,
        )

        return new_state

    @partial(jax.jit, static_argnames=("self",))
    def _mutation_function_pg(
        self,
        controller_params: Genotype,
        descriptor: Descriptor,
        emitter_state: DCGMEEmitterState,
    ) -> Genotype:
        """Apply pg mutation to a policy via multiple steps of gradient descent.
        Args:
            controller_params: a controller, supposed to be a differentiable neural
                network.
            emitter_state: the current state of the emitter, containing among others,
                the replay buffer, the critic.
        Returns:
            the updated params of the neural network.
        """

        # Define new controller optimizer state
        controller_optimizer_state = self._controllers_optimizer.init(
            controller_params
        )

        def scan_train_controller(
            carry: Tuple[DCGMEEmitterState, Genotype], unused: Any
        ) -> Tuple[Tuple[DCGMEEmitterState, Genotype], Any]:
            emitter_state, controller_params, descriptors, controller_optimizer_state = carry
            (
                new_emitter_state,
                new_controller_params,
                new_descriptors,
                new_controller_optimizer_state
            ) = self._train_controller(emitter_state, controller_params, descriptors, controller_optimizer_state,)
            return (new_emitter_state, new_controller_params, new_descriptors, new_controller_optimizer_state), ()

        descriptors = jnp.tile(
            descriptor, (self._config.batch_size, 1))/self._config.max_bd

        (emitter_state, controller_params, descriptors, controller_optimizer_state), _ = jax.lax.scan(
            scan_train_controller,
            (emitter_state, controller_params,
             descriptors, controller_optimizer_state),
            (),
            length=self._config.num_pg_training_steps,
        )

        return controller_params

    @partial(jax.jit, static_argnames=("self",))
    def _train_controller(
        self,
        emitter_state: DCGMEEmitterState,
        controller_params: Params,
        descriptors: Descriptor,
        controller_optimizer_state: optax.OptState,
    ) -> Tuple[DCGMEEmitterState, Params]:
        """Apply one gradient step to a policy (called controllers_params).
        Args:
            emitter_state: current state of the emitter.
            controller_params: parameters corresponding to the weights and bias of
                the neural network that defines the controller.
        Returns:
            The new emitter state and new params of the NN.
        """

        # Sample a batch of transitions in the buffer
        random_key = emitter_state.random_key
        replay_buffer = emitter_state.replay_buffer
        samples, random_key = replay_buffer.sample(
            random_key, sample_size=self._config.batch_size
        )

        # Concatenate states and descriptors and scale rewards
        samples = samples.replace(rewards=similarity(samples.desc, descriptors, self._config.lengthscale)*samples.rewards,
                                  input_desc=descriptors)

        policy_loss, policy_gradient = jax.value_and_grad(self._policy_loss_fn)(
            controller_params,
            emitter_state.critic_params,
            samples,
        )
        # Compute gradient and update policies
        (policy_updates, controller_optimizer_state,) = self._controllers_optimizer.update(
            policy_gradient, controller_optimizer_state
        )
        controller_params = optax.apply_updates(
            controller_params, policy_updates)

        # Create new training state
        new_emitter_state = DCGMEEmitterState(
            critic_params=emitter_state.critic_params,
            critic_optimizer_state=emitter_state.critic_optimizer_state,
            greedy_policy_params=emitter_state.greedy_policy_params,
            greedy_policy_opt_state=emitter_state.greedy_policy_opt_state,
            target_critic_params=emitter_state.target_critic_params,
            target_greedy_policy_params=emitter_state.target_greedy_policy_params,
            random_key=random_key,
            steps=emitter_state.steps,
            replay_buffer=replay_buffer,
        )

        return new_emitter_state, controller_params, descriptors, controller_optimizer_state

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._config.batch_size
