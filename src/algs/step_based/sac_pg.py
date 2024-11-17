import numpy as np

import torch
import torch.optim as optim
import torch.distributions as torchdist

from copy import deepcopy
from itertools import chain

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.utils.replay_memory import ReplayMemory
from mushroom_rl.utils.torch import to_float_tensor

from src.distributions.torchdist_utils import DoubleSidedStandardMaxwell, standard_gaussian_from_standard_dsmaxwell


EPS = 1e-6
LOG_STD_MAX = 3
LOG_STD_MIN = -10


class SACPolicy(Policy):
    """
    Class used to implement the policy used by the Soft Actor-Critic algorithm.
    The policy is a Gaussian policy with diagonal Covariance squashed by a tanh.
    This class implements the compute_action_and_log_prob and the
    compute_action_and_log_prob_t methods, that are fundamental for
    the internals calculations of the SAC algorithm.

    """
    def __init__(self, mu_approximator, sigma_approximator, min_a, max_a):
        """
        Constructor.

        Args:
            mu_approximator (Regressor): a regressor computing mean in given a
                state;
            sigma_approximator (Regressor): a regressor computing the variance
                in given a state;
            min_a (np.ndarray): a vector specifying the minimum action value
                for each component;
            max_a (np.ndarray): a vector specifying the maximum action value
                for each component.

        """
        self._s_dim = mu_approximator.input_shape[0]
        self._a_dim = mu_approximator.output_shape[0]

        self._mu_approximator = mu_approximator
        self._sigma_approximator = sigma_approximator

        self._delta_a = to_float_tensor(.5 * (max_a - min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (max_a + min_a), self.use_cuda)

        use_cuda = self._mu_approximator.model.use_cuda

        if use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._torch_device_policy = torch.device('cuda' if use_cuda else 'cpu')

        self._add_save_attr(
            _mu_approximator='mushroom',
            _sigma_approximator='mushroom',
            _delta_a='torch',
            _central_a='torch'
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.compute_action_a_and_log_prob_a_t(
            state, compute_log_prob=False).detach().cpu().numpy()

    def compute_action_a_and_log_prob_a(self, state):
        """
        Function that samples actions using the reparametrization trick and
        the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_a_and_log_prob_a_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_a_and_log_prob_a_t(self, state, compute_log_prob=True):
        """
        Function that samples actions using the reparametrization trick and,
        optionally, the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log
            probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch
            tensors.

        """
        dist = self.distribution(state)
        a_raw = dist.rsample()
        a = torch.tanh(a_raw)
        a_true = a * self._delta_a + self._central_a

        if compute_log_prob:
            log_prob = dist.log_prob(a_raw).sum(dim=1)
            log_prob -= torch.log(self._delta_a * (1. - a.pow(2)) + EPS).sum(dim=1)
            return a_true, log_prob
        else:
            return a_true

    def compute_a_from_u(self, u):
        a = self._central_a + self._delta_a * torch.tanh(u)
        return a

    def log_prob_a_from_u(self, s, u):
        dist = self.distribution(s)
        log_prob = dist.log_prob(u).sum(dim=1, keepdim=True)
        log_prob -= torch.log(self._delta_a * (1. - torch.tanh(u).pow(2)) + EPS).sum(dim=1, keepdim=True)
        return log_prob

    def draw_action_u_t(self, s):
        dist = self.distribution(s)
        u = dist.rsample()
        return u

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is
                computed.

        Returns:
            The torch distribution for the provided states.

        """
        mu, sigma = self.get_mean_and_std(state, tensor=True)
        return torch.distributions.Normal(mu, sigma)

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray): the set of states to consider.

        Returns:
            The value of the entropy of the policy.

        """

        return torch.mean(self.distribution(state).entropy()).detach().cpu().numpy().item()

    def reset(self):
        pass

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by
                the policy.

        """
        mu_weights = weights[:self._mu_approximator.weights_size]
        sigma_weights = weights[self._mu_approximator.weights_size:]

        self._mu_approximator.set_weights(mu_weights)
        self._sigma_approximator.set_weights(sigma_weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        mu_weights = self._mu_approximator.get_weights()
        sigma_weights = self._sigma_approximator.get_weights()

        return np.concatenate([mu_weights, sigma_weights])

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._mu_approximator.model.use_cuda

    def get_mean_and_std(self, state, tensor=True):
        mu = self._mu_approximator.predict(state, output_tensor=tensor)
        log_sigma = self._sigma_approximator.predict(state, output_tensor=tensor)
        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = log_sigma.exp()
        return mu, sigma

    def mvd_grad_gaussian_mean(self, f, states, actions_u, coupling=True):
        """
        Computes the measure valued derivative wrt the mean of the multivariate Gaussian with diagonal Covariance.
        states shape : (N, Ds)
        actions shape: (N, Da)
        """
        mean, std = self.get_mean_and_std(states)
        diag_std = std

        assert diag_std.shape == actions_u.shape

        # Replicate the second to last dimension of actions
        # (N, Da, Da)
        multiples = [1, self._a_dim, 1]
        base_actions = torch.unsqueeze(actions_u, -2).repeat(*multiples)

        # Sample (NxNaxDa, Da) samples from the positive and negative Univariate Weibull distributions
        weibull = torchdist.weibull.Weibull(scale=np.sqrt(2.), concentration=2.)
        pos_samples_weibull = weibull.sample(actions_u.shape).to(device=self._torch_device_policy)

        if coupling:
            neg_samples_weibull = pos_samples_weibull
        else:
            neg_samples_weibull = weibull.sample(actions_u.shape).to(device=self._torch_device_policy)

        # Build the (N, Da) positive and negative diagonals of the MVD decomposition
        positive_diag = mean + diag_std * pos_samples_weibull
        assert positive_diag.shape == actions_u.shape

        negative_diag = mean - diag_std * neg_samples_weibull
        assert negative_diag.shape == actions_u.shape

        # Set the positive and negative points where to evaluate the objective function.
        # (N, Da, Da)
        # Replace the ith dimension of the actions with the ith entry of the constructed diagonals.
        # Mohamed. S, 2019, Monte Carlo Gradient Estimation in Machine Learning, Ch. 6.2
        positive_samples = base_actions.clone()
        positive_samples.diagonal(dim1=-2, dim2=-1).copy_(positive_diag)
        negative_samples = base_actions.clone()
        negative_samples.diagonal(dim1=-2, dim2=-1).copy_(negative_diag)

        # MVD constant term
        # (N, Da)
        c = np.sqrt(2 * np.pi) * diag_std

        # Compute objective function
        states_repeat = states.repeat(1, self._a_dim).view(-1, self._s_dim)
        positive_samples_reshape = positive_samples.reshape(actions_u.shape[0] * self._a_dim, self._a_dim)
        negative_samples_reshape = negative_samples.reshape(actions_u.shape[0] * self._a_dim, self._a_dim)

        pos_f_samples = f(states_repeat,
                          self.compute_a_from_u(positive_samples_reshape),
                          self.log_prob_a_from_u(states_repeat, positive_samples_reshape)
                          )

        neg_f_samples = f(states_repeat,
                          self.compute_a_from_u(negative_samples_reshape),
                          self.log_prob_a_from_u(states_repeat, negative_samples_reshape)
                          )

        # Gradient batch
        # (N, Da)
        delta_f = pos_f_samples - neg_f_samples
        grad = delta_f.reshape(actions_u.shape[0], self._a_dim) / c
        assert grad.shape == actions_u.shape

        return grad

    def mvd_grad_gaussian_cov(self, f, states, actions_u, coupling=True):
        """
        Computes the measure valued derivative wrt the covariance of the multivariate Gaussian with diagonal Covariance.
        states shape : (N, Ds)
        actions shape: (N, Da)
        """
        mean, std = self.get_mean_and_std(states)
        diag_std = std

        assert diag_std.shape == actions_u.shape

        # Replicate the second to last dimension of actions
        # (N, Da, Da)
        multiples = [1, self._a_dim, 1]
        base_actions = torch.unsqueeze(actions_u, -2).repeat(*multiples)

        # Sample (NxBxDa, Da) samples from the positive and negative Univariate distributions of the decomposition.
        # The positive part is a Double-sided Maxwell M(mu, sigma^2).
        #   M(x; mu, sigma^2) = 1/(sigma*sqrt(2*pi)) * ((x-mu)/sigma)^2 * exp(-1/2*((x-mu)/sigma)^2)
        #   To sample Y from the Double-sided Maxwell M(mu, sigma^2) we can do
        #   X ~ M(0, 1) -> Y = mu + sigma * X
        # The negative part is a Gaussian distribution N(mu, sigma^2).
        #   To sample Y from the Gaussian N(mu, sigma^2) we can do
        #   X ~ N(0, 1) -> Y = mu + sigma * X
        double_sided_maxwell_standard = DoubleSidedStandardMaxwell()
        pos_samples_double_sided_maxwell_standard = double_sided_maxwell_standard.sample(actions_u.shape)

        if coupling:
            # Construct standard Gaussian samples from standard Double-sided Maxwell samples
            neg_samples_gaussian_standard = standard_gaussian_from_standard_dsmaxwell(pos_samples_double_sided_maxwell_standard).to(device=self._torch_device_policy)
        else:
            gaussian_standard = torchdist.normal.Normal(loc=0., scale=1.)
            neg_samples_gaussian_standard = gaussian_standard.sample(actions_u.shape).to(device=self._torch_device_policy)

        pos_samples_double_sided_maxwell_standard = pos_samples_double_sided_maxwell_standard.to(device=self._torch_device_policy)

        # Build the (N, Da) positive and negative diagonals of the MVD decomposition
        positive_diag = mean + diag_std * pos_samples_double_sided_maxwell_standard
        assert positive_diag.shape == actions_u.shape

        negative_diag = mean + diag_std * neg_samples_gaussian_standard
        assert negative_diag.shape == actions_u.shape

        # Set the positive and negative points where to evaluate the objective function.
        # (N, Da, Da)
        # In multivariate Gaussians with diagonal covariance, the univariates are independent.
        # Hence we can replace the ith dimension of the sampled actions with the ith entry of the constructed diagonals.
        # Mohamed. S, 2019, Monte Carlo Gradient Estimation in Machine Learning, Ch. 6.2
        positive_samples = base_actions.clone()
        positive_samples.diagonal(dim1=-2, dim2=-1).copy_(positive_diag)
        negative_samples = base_actions.clone()
        negative_samples.diagonal(dim1=-2, dim2=-1).copy_(negative_diag)

        # MVD constant term
        # (N, Da)
        c = diag_std

        # Compute objective function
        states_repeat = states.repeat(1, self._a_dim).view(-1, self._s_dim)
        positive_samples_reshape = positive_samples.reshape(actions_u.shape[0] * self._a_dim, self._a_dim)
        negative_samples_reshape = negative_samples.reshape(actions_u.shape[0] * self._a_dim, self._a_dim)

        pos_f_samples = f(states_repeat,
                          self.compute_a_from_u(positive_samples_reshape),
                          self.log_prob_a_from_u(states_repeat, positive_samples_reshape)
                          )

        neg_f_samples = f(states_repeat,
                          self.compute_a_from_u(negative_samples_reshape),
                          self.log_prob_a_from_u(states_repeat, negative_samples_reshape)
                          )

        # Gradient batch
        # (N, Da)
        delta_f = pos_f_samples - neg_f_samples
        grad = delta_f.reshape(actions_u.shape[0], self._a_dim) / c
        assert grad.shape == actions_u.shape

        return grad


class SAC_PolicyGradient(DeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    With different Monte Carlo Gradient Estimators: Reparametrization Trick, Score-Function, Measure-Valued Derivative.

    """
    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params,
                 actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau,
                 lr_alpha, target_entropy=None, critic_fit_params=None,
                 mc_gradient_estimator=None):
        """
        Constructor.

        Args:
            actor_mu_params (dict): parameters of the actor mean approximator
                to build;
            actor_sigma_params (dict): parameters of the actor sigm
                approximator to build;
            actor_optimizer (dict): parameters to specify the actor
                optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to
                build;
            batch_size (int): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before
                starting the learning;
            max_replay_size (int): the maximum number of samples in the replay
                memory;
            warmup_transitions (int): number of samples to accumulate in the
                replay memory to start the policy fitting;
            tau (float): value of coefficient for soft updates;
            lr_alpha (float): Learning rate for the entropy coefficient;
            target_entropy (float, None): target entropy for the policy, if
                None a default value is computed ;
            critic_fit_params (dict, None): parameters of the fitting algorithm
                of the critic approximator.

        """
        self._mc_gradient_estimator = mc_gradient_estimator
        self._coupling = False
        if 'coupling' in self._mc_gradient_estimator:
            self._coupling = self._mc_gradient_estimator['coupling']

        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = batch_size
        self._warmup_transitions = warmup_transitions
        self._tau = to_parameter(tau)

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)
        else:
            self._target_entropy = target_entropy

        self._replay_memory = ReplayMemory(initial_replay_size, max_replay_size)

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator,
                                              **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator,
                                                     **target_critic_params)

        actor_mu_approximator = Regressor(TorchApproximator,
                                          **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator,
                                             **actor_sigma_params)

        policy = SACPolicy(actor_mu_approximator,
                           actor_sigma_approximator,
                           mdp_info.action_space.low,
                           mdp_info.action_space.high)

        self._init_target(self._critic_approximator,
                          self._target_critic_approximator)

        self._log_alpha = torch.tensor(0., dtype=torch.float32)

        if policy.use_cuda:
            self._log_alpha = self._log_alpha.cuda().requires_grad_()
        else:
            self._log_alpha.requires_grad_()

        self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters())

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='primitive',
            _warmup_transitions='primitive',
            _tau='primitive',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _log_alpha='torch',
            _alpha_optim='torch'
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset):
        self._replay_memory.add(dataset)
        if self._replay_memory.initialized:
            state, action, reward, next_state, absorbing, _ = \
                self._replay_memory.get(self._batch_size)

            if self._replay_memory.size > self._warmup_transitions:
                ##############################
                # Update ACTOR
                # The loss has two components if the distribution and the function f share parameters W
                # d/dW E_{p(x;W)}[f(x;W)] = \int d/dW[p(x;W)] f(x;W) + \int p(x;W) d/dW[f(x;W)]
                # Ns - state samples
                # Na - action MC samples for each state
                # Ds - state dim
                # Da - action dim

                # (Ns, Ds)
                state_t = torch.tensor(state)
                Ns = state_t.shape[0]
                # (NsxNa, Ds)
                state_repeat_t = state_t.repeat((self._mc_gradient_estimator['n_samples'], 1))

                # (NsxNa, Da)
                action_u_t = self.policy.draw_action_u_t(state_repeat_t)
                action_a_t = self.policy.compute_a_from_u(action_u_t)

                # Reptrick
                if self._mc_gradient_estimator['estimator'] == 'reptrick':
                    log_prob_a_t = self.policy.log_prob_a_from_u(state_repeat_t, action_u_t)
                    loss1 = self._loss_objective(state_repeat_t, action_a_t, log_prob_a_t).mean()
                    loss = loss1

                elif self._mc_gradient_estimator['estimator'] == 'sf':
                    log_prob_a_t = self.policy.log_prob_a_from_u(state_repeat_t, action_u_t.detach())

                    loss1 = self._loss_sf(state_repeat_t, action_a_t, log_prob_a_t).mean()
                    # For the loss \int p(x;W) d/dW[f(x;W)], it is enough to use one action sample per state
                    loss2 = self._loss_additional(log_prob_a_t[:Ns]).mean()
                    loss = loss1 + loss2

                elif self._mc_gradient_estimator['estimator'] == 'mvd':
                    log_prob_a_t = self.policy.log_prob_a_from_u(state_repeat_t, action_u_t.detach())

                    loss1 = self._loss_mvd(state_repeat_t, action_u_t).mean()
                    # For the loss \int p(x;W) d/dW[f(x;W)], it is enough to use one action sample per state
                    loss2 = self._loss_additional(log_prob_a_t[:Ns]).mean()
                    loss = loss1 + loss2

                else:
                    raise NotImplementedError

                self._optimize_actor_parameters(loss)
                self._update_alpha(log_prob_a_t[:Ns].detach())

            ##############################
            # Update CRITIC
            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q,
                                          **self._critic_fit_params)

            self._update_target(self._critic_approximator,
                                self._target_critic_approximator)

    def _loss_objective(self, state, action_a, log_prob_a):
        q_0 = self._critic_approximator(state, action_a,
                                        output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_a,
                                        output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return self._alpha * log_prob_a - q.view(-1, 1)

    def _loss_additional(self, log_prob_a_t):
        return self._alpha * log_prob_a_t

    def _loss_sf(self, state, action_a, log_prob_a_t):
        loss = log_prob_a_t * self._loss_objective(state, action_a, log_prob_a_t).detach()

        return loss

    def _loss_mvd(self, states, actions_u):
        """
        Builds the loss function for gradient computation with measure value derivatives.
        The gradient is taken wrt the distributional parameters of a
        Multivariate Gaussian with Diagonal Covariance (mean and covariance).
        """
        mean, std = self.policy.get_mean_and_std(states)
        diag_std = std

        assert diag_std.shape == actions_u.shape

        # Compute gradient wrt mean
        grad_mean = self.policy.mvd_grad_gaussian_mean(self._loss_objective, states, actions_u, coupling=self._coupling)

        # Compute gradient wrt covariance
        grad_cov = self.policy.mvd_grad_gaussian_cov(self._loss_objective, states, actions_u, coupling=self._coupling)

        # Construct the surrogate loss.
        # Here we still backpropagate through the mean and covariance, because they are parameterized
        surrogate_loss = grad_mean.detach() * mean
        surrogate_loss += grad_cov.detach() * diag_std

        # (N, Da)
        assert surrogate_loss.shape == actions_u.shape

        # The total derivative is the sum of the partial derivatives wrt each parameter.
        # Sum along the action dimension.
        loss = surrogate_loss.sum(dim=-1)

        return loss

    def _update_alpha(self, log_prob):
        alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be
                evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in
                ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the
            action returned by the actor.

        """
        a, log_prob_next = self.policy.compute_action_a_and_log_prob_a(next_state)

        q = self._target_critic_approximator.predict(
            next_state, a, prediction='min') - self._alpha_np * log_prob_next
        q *= 1 - absorbing

        return q

    def _post_load(self):
        if self._optimizer is not None:
            self._parameters = list(
                chain(self.policy._mu_approximator.model.network.parameters(),
                      self.policy._sigma_approximator.model.network.parameters()
                )
            )

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()


