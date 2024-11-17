import time

import numpy as np

from src.utils.dtype_utils import NP_FLOAT_DTYPE
from src.utils.replay_memory import ReplayMemory
from tqdm import tqdm

import torch

from mushroom_rl.algorithms.agent import Agent
from mushroom_rl.utils.torch import to_float_tensor, get_gradient
from mushroom_rl.utils.minibatches import minibatch_generator
from mushroom_rl.utils.dataset import parse_dataset, compute_J


class Oracle_LQR_PolicyGradient(Agent):
    """
    Policy Gradient with Oracle Critics Q and V, for stochastic policies.

    """
    def __init__(self, mdp_info,
                 policy, actor_optimizer,
                 q_function,
                 v_function,
                 n_episodes_learn=1,
                 n_epochs_policy=5,
                 mc_gradient_estimator=None,
                 true_gradient=None,
                 quiet=False,
                 ):
        """
        Constructor.

        Args:
            ...

        """
        self._Q = q_function
        self._V = v_function

        self._true_gradient = true_gradient

        self._actor_optimizer = actor_optimizer['class'](policy.parameters(), **actor_optimizer['params'])
        self._n_episodes_learn = n_episodes_learn
        self._n_epochs_policy = n_epochs_policy

        self._mc_gradient_estimator = mc_gradient_estimator
        self._mc_gradient_estimator_method = mc_gradient_estimator['estimator']
        self._mc_samples_grad = self._mc_gradient_estimator['n_samples']
        self._coupling = False
        if 'coupling' in self._mc_gradient_estimator:
            self._coupling = self._mc_gradient_estimator['coupling']

        self._quiet = quiet

        self._torch_device_policy = torch.device('cuda' if policy.use_cuda else 'cpu')

        self._iter = 0

        self._true_grads = []
        self._grads = []
        self._policy_params = []

        self._add_save_attr(
            _critic_fit_params='pickle',
            _n_epochs_policy='primitive',
            _actor_optimizer='torch',
            _Q_approx='mushroom',
            _quiet='primitive',
            _iter='primitive',
            _q_target='numpy'
        )

        super().__init__(mdp_info, policy, None)

        self._policy_params.append([p.clone().cpu().detach().numpy() for p in self.policy.parameters()])

    def fit(self, dataset):
        states, actions, rewards, states_next, absorbing, last = parse_dataset(dataset)
        states = states.astype(NP_FLOAT_DTYPE)
        actions = actions.astype(NP_FLOAT_DTYPE)
        rewards = rewards.astype(NP_FLOAT_DTYPE)
        states_next = states_next.astype(NP_FLOAT_DTYPE)
        absorbing = absorbing.astype(NP_FLOAT_DTYPE)
        last = last.astype(NP_FLOAT_DTYPE)
        discounts = np.ones((states.shape[0], 1))

        t = 0
        for i in range(discounts.shape[0]):
            discounts[i] = self.mdp_info.gamma**t
            t += 1
            if last[i] == 1:
                t = 0

        # Set current policy
        K = -1 * self.policy._mu.get_weights()
        K = K.reshape(self.policy._mu.output_shape[0], self.policy._mu.input_shape[0])
        Sigma = self.policy.get_cov(output_tensor=False)
        self._V.set_params(K, Sigma)
        self._Q.set_params(K, Sigma)

        # Policy Gradient
        self._update_policy(states, discounts)

        self._print_info(dataset)
        self._iter += 1

    def _objective_mvd(self, sa):
        return self._Q(sa)

    def _objective_sf(self, sa):
        s = sa[:, :self.mdp_info.observation_space.shape[0]]
        return self._Q(sa) - self._V(s)

    def _objective_reptrick(self, sa):
        return self._Q(sa)

    def _update_policy(self, states, discounts):
        for epoch in range(self._n_epochs_policy):
            states_i = states
            discounts_i = discounts

            self._actor_optimizer.zero_grad()

            states_i_t_repeat = to_float_tensor(states_i).to(device=self._torch_device_policy).repeat(self._mc_samples_grad, 1)
            discounts_i_t = to_float_tensor(discounts_i).to(device=self._torch_device_policy)
            actions_i_t_repeat = self.policy.draw_action_t(states_i_t_repeat, reparametrize=True)

            if self._mc_gradient_estimator_method == 'mvd':
                loss = -1. * self.policy.surrogate_opt_mvd(states_i_t_repeat, actions_i_t_repeat, self._objective_mvd,
                                                           self._coupling,
                                                           discounts=discounts_i_t, mc_samples_grad=self._mc_samples_grad)

            elif self._mc_gradient_estimator_method == 'sf':
                loss = -1. * self.policy.surrogate_opt_sf(states_i_t_repeat, actions_i_t_repeat, self._objective_sf,
                                                          discounts_i_t=discounts_i_t, mc_samples_grad=self._mc_samples_grad)

            elif self._mc_gradient_estimator_method == 'reptrick':
                loss = -1. * self.policy.surrogate_opt_reptrick(states_i_t_repeat, actions_i_t_repeat,
                                                                self._objective_reptrick,
                                                                discounts_i_t=discounts_i_t,
                                                                mc_samples_grad=self._mc_samples_grad)

            else:
                raise NotImplementedError

            loss = loss / self._n_episodes_learn
            loss.backward()

            self._true_grads.append(self._true_gradient(self.policy))
            self._grads.append(get_gradient(self.policy.parameters()).clone().cpu().detach().numpy())

            self._actor_optimizer.step()

            self._policy_params.append([p.clone().cpu().detach().numpy() for p in self.policy.parameters()])

    def _print_info(self, dataset):
        if not self._quiet:
            mean_rwd = np.mean(compute_J(dataset))
            print(f"\t--->iter: {self._iter}, mean reward: {mean_rwd:.3f}")

    def get_grads(self):
        return self._true_grads, self._grads

    def get_policy_params(self):
        return self._policy_params
