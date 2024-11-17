import time

import numpy as np

from src.utils.dtype_utils import NP_FLOAT_DTYPE
from src.utils.replay_memory import ReplayMemory

import torch

from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.dataset import parse_dataset, compute_J


class RegressionTrees_PolicyGradient(Agent):
    """
    Policy Gradient with Regression Trees.

    """
    def __init__(self, mdp_info,
                 policy, actor_optimizer,
                 critic,
                 n_iters_critic=10,
                 critic_fit_params=None,
                 n_epochs_policy=5, batch_size_policy=16,
                 mc_samples_action_next=1,
                 mc_gradient_estimator=None,
                 use_replay_memory=False,
                 max_replay_memory_size=100000,
                 replay_memory_samples=1000,
                 quiet=False):
        """
        Constructor.

        Args:
            ...

        """
        self._Q_approx = critic
        self._n_iters_critic = n_iters_critic
        self._mc_samples_action_next = mc_samples_action_next
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._actor_optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])
        self._n_epochs_policy = n_epochs_policy
        self._batch_size_policy = batch_size_policy

        self._use_replay_memory = use_replay_memory
        self._replay_memory_samples = replay_memory_samples
        self._replay_memory = ReplayMemory(replay_memory_samples, max_replay_memory_size)

        self._mc_gradient_estimator = mc_gradient_estimator
        self._mc_gradient_estimator_method = mc_gradient_estimator['estimator']
        self._mc_samples_grad = self._mc_gradient_estimator['n_samples']
        self._coupling = False
        if 'coupling' in self._mc_gradient_estimator:
            self._coupling = self._mc_gradient_estimator['coupling']

        self._quiet = quiet

        self._torch_device_policy = torch.device('cuda' if policy.use_cuda else 'cpu')

        self._iter = 0

        self._add_save_attr(
            _critic_fit_params='pickle',
            _n_epochs_policy='primitive',
            _batch_size_policy='primitive',
            _actor_optimizer='torch',
            _Q_approx='mushroom',
            _quiet='primitive',
            _iter='primitive',
            _q_target='numpy'
        )

        super().__init__(mdp_info, policy, None)

    def fit(self, dataset):
        if self._use_replay_memory:
            self._replay_memory.add(dataset)

        states, actions, rewards, states_next, absorbing, last = parse_dataset(dataset)
        states = states.astype(NP_FLOAT_DTYPE)
        actions = actions.astype(NP_FLOAT_DTYPE)
        rewards = rewards.astype(NP_FLOAT_DTYPE)
        states_next = states_next.astype(NP_FLOAT_DTYPE)
        absorbing = absorbing.astype(NP_FLOAT_DTYPE)

        states_q_fit = states
        actions_q_fit = actions
        rewards_q_fit = rewards
        states_next_q_fit = states_next
        absorbing_q_fit = absorbing

        if self._use_replay_memory and self._replay_memory.initialized:
            # Sample from replay memory
            states_replay, actions_replay, rewards_replay, states_next_replay, absorbings_replay, _ = \
                self._replay_memory.get(self._replay_memory_samples)
            states_replay = states_replay.astype(NP_FLOAT_DTYPE)
            actions_replay = actions_replay.astype(NP_FLOAT_DTYPE)
            rewards_replay = rewards_replay.astype(NP_FLOAT_DTYPE)
            states_next_replay = states_next_replay.astype(NP_FLOAT_DTYPE)
            absorbings_replay = absorbings_replay.astype(NP_FLOAT_DTYPE)

            states_q_fit = np.concatenate((states_q_fit, states_replay), axis=0)
            actions_q_fit = np.concatenate((actions_q_fit, actions_replay), axis=0)
            rewards_q_fit = np.concatenate((rewards_q_fit, rewards_replay), axis=0)
            states_next_q_fit = np.concatenate((states_next_q_fit, states_next_replay), axis=0)
            absorbing_q_fit = np.concatenate((absorbing_q_fit, absorbings_replay), axis=0)

        # Fit the Q-function with Approximate Policy Evaluation
        st = time.time()
        states_actions_q_fit = np.concatenate((states_q_fit, actions_q_fit), axis=1)
        states_next_q_fit_repeat = np.repeat(states_next_q_fit, self._mc_samples_action_next, axis=0)
        actions_next_repeat = self.policy.draw_action(states_next_q_fit_repeat)
        for n in range(self._n_iters_critic):
            states_actions_next = np.concatenate((states_next_q_fit_repeat, actions_next_repeat), axis=1)
            if self._iter == 0:
                q_target = rewards_q_fit
            else:
                q_next_expected = self._Q_approx.predict(states_actions_next).reshape((states_next_q_fit.shape[0], -1)).mean(axis=1)
                q_target = rewards_q_fit + (1-absorbing_q_fit) * self.mdp_info.gamma * q_next_expected

            self._Q_approx.fit(states_actions_q_fit, q_target, **self._critic_fit_params)

            # Fit statistics
            if n == 0 or n == self._n_iters_critic - 1:
                self._print_fit_info(n, states_actions_q_fit, states_next_q_fit, states_actions_next, rewards_q_fit, q_target)

        print(f"q fit time: {time.time() - st:.3f}")

        # Policy Gradient
        self._update_policy(states)

        self._print_info(dataset)
        self._iter += 1

    def _update_policy(self, states):
        st = time.time()
        for epoch in range(self._n_epochs_policy):
            for states_i in minibatch_generator(self._batch_size_policy, states):
                self._actor_optimizer.zero_grad()

                states_i_t_repeat = to_float_tensor(states_i).squeeze().to(device=self._torch_device_policy).\
                    repeat(self._mc_samples_grad, 1)
                actions_i_t_repeat = self.policy.draw_action_t(states_i_t_repeat)

                if self._mc_gradient_estimator_method == 'mvd':
                    loss = -1. * self.policy.surrogate_opt_mvd(states_i_t_repeat, actions_i_t_repeat, self._Q_approx,
                                                               self._coupling)
                elif self._mc_gradient_estimator_method == 'sf':
                    loss = -1. * self.policy.surrogate_opt_sf(states_i_t_repeat, actions_i_t_repeat, self._Q_approx)
                else:
                    raise NotImplementedError

                loss.backward()
                self._actor_optimizer.step()
        print(f"policy update time: {time.time() - st:.3f}")

    def _print_fit_info(self, n, states_actions, states_next, states_actions_next, rewards, q_target):
        if not self._quiet:
            pred = self._Q_approx.predict(states_actions)
            value_error_mean = np.square(pred - q_target).mean()
            td_error_mean = np.square(
                pred - (rewards + self.mdp_info.gamma *
                        self._Q_approx.predict(states_actions_next).reshape((states_next.shape[0], -1)).mean(axis=1))).mean()

            print(f"\tn: {n} - TD:{td_error_mean:.3f}, VE:{value_error_mean:.3f}")

    def _print_info(self, dataset):
        if not self._quiet:
            mean_rwd = np.mean(compute_J(dataset))
            print(f"\t--->iter: {self._iter}, mean reward: {mean_rwd:.3f}")

