import sys
import time
import warnings
from typing import Any, ClassVar, Dict, Generic, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.pytree_dataclass import tree_index, tree_map
from stable_baselines3.common.recurrent.buffers import (
    RecurrentRolloutBuffer,
    SamplingType,
)
from stable_baselines3.common.recurrent.policies import BaseRecurrentActorCriticPolicy
from stable_baselines3.common.recurrent.torch_layers import RecurrentState
from stable_baselines3.common.recurrent.type_aliases import RecurrentRolloutBufferData
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    Schedule,
    TorchGymObs,
    non_null,
)
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    safe_mean,
)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.util import obs_as_tensor
from stable_baselines3.ppo_recurrent.policies import (
    CnnLstmPolicy,
    MlpLstmPolicy,
    MultiInputLstmPolicy,
)

SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")


class RecurrentPPO(OnPolicyAlgorithm, Generic[RecurrentState]):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    with support for recurrent policies (LSTM).

    Based on the original Stable Baselines 3 implementation.

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
        "MlpPolicy": MlpLstmPolicy,
        "CnnPolicy": CnnLstmPolicy,
        "MultiInputPolicy": MultiInputLstmPolicy,
    }

    policy: BaseRecurrentActorCriticPolicy[RecurrentState]
    policy_class: Type[BaseRecurrentActorCriticPolicy[RecurrentState]]
    rollout_buffer: RecurrentRolloutBuffer

    def __init__(
        self,
        policy: Union[str, Type[BaseRecurrentActorCriticPolicy[RecurrentState]]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 128,
        batch_envs: int = 128,
        batch_time: Optional[int] = None,
        sampling_type: SamplingType = SamplingType.CLASSIC,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: Optional[float] = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        if batch_time is None:
            batch_time = self.n_steps
        self.sampling_type = sampling_type
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_envs * batch_time > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            num_envs = self.env.num_envs
            assert (
                num_envs > 1 or batch_time > 1 or (not normalize_advantage)
            ), f"`num_envs` or `batch_time` must be greater than 1. Currently num_envs={num_envs} and batch_time={batch_time}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            if (truncated_batch_size := num_envs % batch_envs) > 0:
                untruncated_batches = num_envs // batch_envs
                warnings.warn(
                    f"You have specified an environment mini-batch size of {batch_envs},"
                    f" but because the `RecurrentRolloutBuffer` has `n_envs = {self.env.num_envs}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {truncated_batch_size}\n"
                    f"We recommend using a `batch_envs` that is a factor of `n_envs`.\n"
                    f"Info: (n_envs={self.env.num_envs})"
                )

            if (truncated_batch_size := self.n_steps % batch_time) > 0:
                untruncated_batches = self.n_steps // batch_time
                warnings.warn(
                    f"You have specified a time mini-batch size of {batch_time},"
                    f" but because the `RecurrentRolloutBuffer` has `n_steps = {self.n_steps}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {truncated_batch_size}\n"
                    f"We recommend using a `batch_time` that is a factor of `n_steps`.\n"
                    f"Info: (n_envs={self.n_steps})"
                )

        self.batch_envs = batch_envs
        self.batch_time = batch_time
        self.n_epochs = n_epochs
        self.clip_range: Schedule = clip_range  # type: ignore
        self.clip_range_vf: Schedule = clip_range_vf  # type: ignore
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self._last_lstm_states: Optional[RecurrentState] = None

        if _init_setup_model:
            self._setup_model()

    def think_for_n_steps(
        self, n_steps: int, obs_tensor: TorchGymObs, lstm_states: Optional[RecurrentState], episode_starts: th.Tensor
    ) -> RecurrentState:
        if lstm_states is None:
            out = self.policy.recurrent_initial_state(episode_starts.size(0), device=self.device)
            lstm_states = out

        if not episode_starts.any() or n_steps == 0:
            return lstm_states
        # ignore because TorchGymObs and TensorTree do not match
        obs_for_start_envs: TorchGymObs = tree_index(obs_tensor, (episode_starts,))  # type: ignore[type-var]
        lstm_states_for_start_envs = tree_index(lstm_states, (slice(None), episode_starts))

        reset_all = th.ones(int(episode_starts.sum().item()), device=self.device, dtype=th.bool)
        do_not_reset = ~reset_all
        for step_i in range(n_steps):
            _, _, _, lstm_states_for_start_envs = self.policy.forward(
                obs_for_start_envs,
                lstm_states_for_start_envs,
                reset_all if step_i == 0 else do_not_reset,
            )

        def _set_thinking(x, y) -> th.Tensor:
            x = x.clone()  # Don't overwrite previous tensor
            x[:, episode_starts] = y
            return x

        lstm_states = tree_map(_set_thinking, lstm_states, lstm_states_for_start_envs)
        return lstm_states

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # if not isinstance(self.policy, RecurrentActorCriticPolicy):
        #     raise ValueError("Policy must subclass RecurrentActorCriticPolicy")

        hidden_state_example: RecurrentState = self.policy.recurrent_initial_state(n_envs=self.n_envs, device=self.device)

        self.rollout_buffer = RecurrentRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_example=hidden_state_example,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            sampling_type=self.sampling_type,
        )
        self._last_lstm_states = tree_map(lambda x: th.zeros_like(x, memory_format=th.contiguous_format), hidden_state_example)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    @th.no_grad()
    def collect_rollouts(  # type: ignore[override]
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RecurrentRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert isinstance(rollout_buffer, RecurrentRolloutBuffer), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        self._last_episode_starts = non_null(self._last_episode_starts).to(self.device)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = non_null(self._last_lstm_states)

        all_rollout_rewards = []
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = non_null(self._last_episode_starts)
                actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)
                lstm_states = tree_map(th.clone, lstm_states)

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = th.clip(
                    actions, th.as_tensor(self.action_space.low).to(actions), th.as_tensor(self.action_space.high).to(actions)
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            all_rollout_rewards.append(rewards.mean().item())

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = tree_map(
                            lambda x: x[:, idx : idx + 1, :].contiguous(),  # noqa: B023  ( idx not captured by function )
                            lstm_states,
                        )
                        episode_starts = th.tensor([False], dtype=th.bool, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[
                            0
                        ].squeeze()
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                RecurrentRolloutBufferData(
                    self._last_obs,
                    actions,
                    rewards,
                    non_null(self._last_episode_starts),
                    values.squeeze(-1),
                    log_probs,
                    hidden_states=non_null(self._last_lstm_states),
                )
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones.to(self.device)
            self._last_lstm_states = lstm_states

        self.logger.record("train/reward", np.mean(all_rollout_rewards))

        # Compute value for the last timestep
        dones = episode_starts = th.as_tensor(dones).to(dtype=th.bool, device=self.device)
        values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(batch_time=self.batch_time, batch_envs=self.batch_envs):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.squeeze(-1)

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_envs * self.batch_time)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,  # type: ignore[arg-type]
                    actions,
                    tree_map(th.clone, rollout_data.hidden_states),  # type: ignore[arg-type]
                    rollout_data.episode_starts,
                )

                values = values.squeeze(-1)
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                with th.no_grad():
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).item()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                if self.max_grad_norm is not None:
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        self.policy.optimizer.zero_grad(set_to_none=True)  # Free gradients until the next call to .train()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRecurrentPPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                non_null(self.env), callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                ep_info_buffer = non_null(self.ep_info_buffer)
                if len(ep_info_buffer) > 0 and len(ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self
