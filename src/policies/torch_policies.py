import torch
import torch.distributions as torchdist
import numpy as np

from mushroom_rl.utils.torch import to_float_tensor

from src.distributions.torchdist_utils import DoubleSidedStandardMaxwell, standard_gaussian_from_standard_dsmaxwell
from src.utils.dtype_utils import TORCH_FLOAT_DTYPE
from src.utils.torch_utils import detach_and_convert_numpy, detach_tensors
from src.policies.torch_policy import GaussianTorchPolicy


class GaussianTorchPolicyExtended(GaussianTorchPolicy):
    """
    Extends GaussianTorchPolicy to use Measure-Valued Derivatives.
    """

    def __init__(self, network, input_shape, output_shape, std_0=1., trainable_std=True,
                 use_cuda=False, log_std_min=-10, log_std_max=6, **params):

        super().__init__(network, input_shape, output_shape, std_0, trainable_std, use_cuda, **params)

        self._s_dim = input_shape[0]
        self._a_dim = output_shape[0]

        self._mvd_params_dim = self._a_dim + self._a_dim  # mu + log_sigma

        self._torch_device_policy = torch.device('cuda' if use_cuda else 'cpu')

        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self._eps = 1e-6

    def draw_action_t(self, state, reparametrize=False):
        return super(GaussianTorchPolicyExtended, self).draw_action_t(state, reparametrize)

    def distribution_t(self, state):
        mu, sigma = self.get_mean_and_covariance(state)
        return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)

    def get_mean(self, state, output_tensor=True):
        return self._mu(state, output_tensor=output_tensor)

    def get_std(self, output_tensor=True):
        log_sigma = torch.clamp(self._log_sigma, self._log_std_min, self._log_std_max)
        if output_tensor:
            return torch.exp(log_sigma)
        return torch.exp(log_sigma).detach().cpu().numpy()

    def get_cov(self, output_tensor=True):
        log_sigma = torch.clamp(self._log_sigma, self._log_std_min, self._log_std_max)
        if output_tensor:
            return torch.exp(2 * log_sigma)
        return np.atleast_2d(torch.exp(2 * log_sigma).detach().cpu().numpy())

    def get_mean_and_std(self, state, output_tensor=True, match_num_states=False):
        mean = self.get_mean(state, output_tensor)
        std = self.get_std(output_tensor)
        if match_num_states:
            std = std.repeat((state.shape[0], 1))
        return mean, std

    def get_mean_and_covariance(self, state, output_tensor=True):
        mean = self.get_mean(state, output_tensor)
        log_sigma = torch.clamp(self._log_sigma, self._log_std_min, self._log_std_max)
        if not output_tensor:
            cov = torch.diag(torch.exp(2 * log_sigma)).detach().cpu().numpy()
        else:
            cov = torch.diag(torch.exp(2 * log_sigma))
        return mean, cov

    def generate_mvd_theta_mean(self, states, actions, coupling=False):
        """
        Generates the positive and negative actions for the measure-valued derivative wrt the mean at states.
        states shape : (N, Ds)
        actions shape: (N, Da)
        """
        mean, std = self.get_mean_and_std(states, output_tensor=True, match_num_states=True)
        diag_std = std

        assert diag_std.shape == actions.shape

        # Replicate the second to last dimension of actions
        # (N, Da, Da)
        multiples = [1, self._a_dim, 1]
        base_actions = torch.unsqueeze(actions, -2).repeat(*multiples)

        # Sample (NxNaxDa, Da) samples from the positive and negative Univariate Weibull distributions
        weibull = torchdist.weibull.Weibull(scale=np.sqrt(2.), concentration=2.)
        pos_samples_weibull = weibull.sample(actions.shape).to(device=self._torch_device_policy)

        if coupling:
            neg_samples_weibull = pos_samples_weibull
        else:
            neg_samples_weibull = weibull.sample(actions.shape).to(device=self._torch_device_policy)

        # Build the (N, Da) positive and negative diagonals of the MVD decomposition
        pos_diag = mean + diag_std * pos_samples_weibull
        assert pos_diag.shape == actions.shape

        neg_diag = mean - diag_std * neg_samples_weibull
        assert neg_diag.shape == actions.shape

        # Set the positive and negative points where to evaluate the objective function.
        # (N, Da, Da)
        # Replace the ith dimension of the actions with the ith entry of the constructed diagonals.
        # Mohamed. S, 2019, Monte Carlo Gradient Estimation in Machine Learning, Ch. 6.2
        pos_action_samples = base_actions.clone()
        pos_action_samples.diagonal(dim1=-2, dim2=-1).copy_(pos_diag)
        neg_action_samples = base_actions.clone()
        neg_action_samples.diagonal(dim1=-2, dim2=-1).copy_(neg_diag)

        # MVD constant term
        # (N, Da)
        c = 1. / (np.sqrt(2 * np.pi) * diag_std)

        pos_action_samples = pos_action_samples.reshape(actions.shape[0] * self._a_dim, self._a_dim)
        neg_action_samples = neg_action_samples.reshape(actions.shape[0] * self._a_dim, self._a_dim)

        return pos_action_samples, neg_action_samples, c

    def generate_mvd_theta_std(self, states, actions, coupling=False):
        """
        Generates the positive and negative actions for the measure-valued derivative wrt the std at states.
        states shape : (N, Ds)
        actions shape: (N, Da)
        """
        mean, std = self.get_mean_and_std(states, output_tensor=True, match_num_states=True)
        diag_std = std

        assert diag_std.shape == actions.shape

        # Replicate the second to last dimension of actions
        # (N, Da, Da)
        multiples = [1, self._a_dim, 1]
        base_actions = torch.unsqueeze(actions, -2).repeat(*multiples)

        # Sample (NxBxDa, Da) samples from the positive and negative Univariate distributions of the decomposition.
        # The positive part is a Double-sided Maxwell M(mu, sigma^2).
        #   M(x; mu, sigma^2) = 1/(sigma*sqrt(2*pi)) * ((x-mu)/sigma)^2 * exp(-1/2*((x-mu)/sigma)^2)
        #   To sample Y from the Double-sided Maxwell M(mu, sigma^2) we can do
        #   X ~ M(0, 1) -> Y = mu + sigma * X
        # The negative part is a Gaussian distribution N(mu, sigma^2).
        #   To sample Y from the Gaussian N(mu, sigma^2) we can do
        #   X ~ N(0, 1) -> Y = mu + sigma * X
        double_sided_maxwell_standard = DoubleSidedStandardMaxwell()
        pos_samples_double_sided_maxwell_standard = double_sided_maxwell_standard.sample(actions.shape)

        if coupling:
            # Construct standard Gaussian samples from standard Double-sided Maxwell samples
            neg_samples_gaussian_standard = standard_gaussian_from_standard_dsmaxwell(
                pos_samples_double_sided_maxwell_standard).to(device=self._torch_device_policy)
        else:
            gaussian_standard = torchdist.normal.Normal(loc=0., scale=1.)
            neg_samples_gaussian_standard = gaussian_standard.sample(actions.shape).to(device=self._torch_device_policy)

        pos_samples_double_sided_maxwell_standard = pos_samples_double_sided_maxwell_standard.to(
            device=self._torch_device_policy)

        # Build the (N, Da) positive and negative diagonals of the MVD decomposition
        pos_diag = mean + diag_std * pos_samples_double_sided_maxwell_standard
        assert pos_diag.shape == actions.shape

        neg_diag = mean + diag_std * neg_samples_gaussian_standard
        assert neg_diag.shape == actions.shape

        # Set the positive and negative points where to evaluate the objective function.
        # (N, Da, Da)
        # In multivariate Gaussians with diagonal covariance, the univariates are independent.
        # Hence we can replace the ith dimension of the sampled actions with the ith entry of the constructed diagonals.
        # Mohamed. S, 2019, Monte Carlo Gradient Estimation in Machine Learning, Ch. 6.2
        pos_action_samples = base_actions.clone()
        pos_action_samples.diagonal(dim1=-2, dim2=-1).copy_(pos_diag)
        neg_action_samples = base_actions.clone()
        neg_action_samples.diagonal(dim1=-2, dim2=-1).copy_(neg_diag)

        # MVD constant term
        # (N, Da)
        c = 1. / diag_std

        pos_action_samples = pos_action_samples.reshape(actions.shape[0] * self._a_dim, self._a_dim)
        neg_action_samples = neg_action_samples.reshape(actions.shape[0] * self._a_dim, self._a_dim)

        return pos_action_samples, neg_action_samples, c

    def mvd_grad_gaussian_mean(self, f, states, actions, coupling=False,
                               transform_action=lambda x: x, transform_state=lambda x: x):
        """
        Computes the measure-valued derivative wrt the mean of the multivariate Gaussian with diagonal Covariance.
        states shape : (N, Ds)
        actions shape: (N, Da)
        """
        pos_action_samples, neg_action_samples, c = self.generate_mvd_theta_mean(states, actions, coupling)

        # Compute objective function
        states_transformed = transform_state(states.repeat(1, self._a_dim).view(-1, self._s_dim))
        if not isinstance(states_transformed, torch.Tensor):
            states_transformed = torch.tensor(states_transformed)
        states_repeat = states_transformed.to(TORCH_FLOAT_DTYPE)
        pos_action_samples = transform_action(pos_action_samples)
        neg_action_samples = transform_action(neg_action_samples)

        states_actions_pos = torch.cat((states_repeat, pos_action_samples), dim=1)
        states_actions_neg = torch.cat((states_repeat, neg_action_samples), dim=1)

        try:
            if f.model.__module__.startswith(('sklearn', 'skmultiflow', 'cuml')):
                states_actions_pos = states_actions_pos.detach().cpu().numpy()
                states_actions_neg = states_actions_neg.detach().cpu().numpy()
        except AttributeError:
            states_actions_pos = states_actions_pos.detach().cpu().numpy()
            states_actions_neg = states_actions_neg.detach().cpu().numpy()

        pos_f_samples = f(states_actions_pos)
        neg_f_samples = f(states_actions_neg)
        if not isinstance(pos_f_samples, torch.Tensor):
            pos_f_samples = torch.tensor(pos_f_samples)
            neg_f_samples = torch.tensor(neg_f_samples)
        pos_f_samples = pos_f_samples.to(device=self._torch_device_policy)
        neg_f_samples = neg_f_samples.to(device=self._torch_device_policy)

        # Gradient batch
        # (N, Da)
        delta_f = pos_f_samples - neg_f_samples
        grad = delta_f.reshape(actions.shape[0], self._a_dim) * c
        assert grad.shape == actions.shape

        return grad

    def mvd_grad_gaussian_std(self, f, states, actions, coupling=False,
                              transform_action=lambda x: x, transform_state=lambda x: x):
        """
        Computes the measure valued derivative wrt the std of the multivariate Gaussian with diagonal Covariance.
        states shape : (N, Ds)
        actions shape: (N, Da)
        """
        pos_action_samples, neg_action_samples, c = self.generate_mvd_theta_std(states, actions, coupling)

        # Compute objective function
        states_transformed = transform_state(states.repeat(1, self._a_dim).view(-1, self._s_dim))
        if not isinstance(states_transformed, torch.Tensor):
            states_transformed = torch.tensor(states_transformed)
        states_repeat = states_transformed.to(TORCH_FLOAT_DTYPE)
        pos_action_samples = transform_action(pos_action_samples)
        neg_action_samples = transform_action(neg_action_samples)

        states_actions_pos = torch.cat((states_repeat, pos_action_samples), dim=1)
        states_actions_neg = torch.cat((states_repeat, neg_action_samples), dim=1)

        try:
            if f.model.__module__.startswith(('sklearn', 'skmultiflow', 'cuml')):
                states_actions_pos = states_actions_pos.detach().cpu().numpy()
                states_actions_neg = states_actions_neg.detach().cpu().numpy()
        except AttributeError:
            states_actions_pos = states_actions_pos.detach().cpu().numpy()
            states_actions_neg = states_actions_neg.detach().cpu().numpy()

        pos_f_samples = f(states_actions_pos)
        neg_f_samples = f(states_actions_neg)
        if not isinstance(pos_f_samples, torch.Tensor):
            pos_f_samples = torch.tensor(pos_f_samples)
            neg_f_samples = torch.tensor(neg_f_samples)
        pos_f_samples = pos_f_samples.to(device=self._torch_device_policy)
        neg_f_samples = neg_f_samples.to(device=self._torch_device_policy)

        # Gradient batch
        # (N, Da)
        delta_f = pos_f_samples - neg_f_samples
        grad = delta_f.reshape(actions.shape[0], self._a_dim) * c
        assert grad.shape == actions.shape

        return grad

    def surrogate_opt_mvd(self, states, actions, f, coupling=False,
                          transform_action=lambda x: x, transform_state=lambda x: x,
                          discounts=None, mc_samples_grad=1):
        """
        Builds the surrogate optimization function to maximize, with gradient computation for measure value derivatives.
        The gradient is taken wrt the distributional parameters of a
        Multivariate Gaussian with Diagonal Covariance (mean and std).
        """
        mean, std = self.get_mean_and_std(states, output_tensor=True, match_num_states=True)
        diag_std = std

        assert diag_std.shape == actions.shape

        # Compute gradient wrt mean
        grad_mean = self.mvd_grad_gaussian_mean(f, states, actions, coupling, transform_action, transform_state)

        # Compute gradient wrt std
        grad_std = self.mvd_grad_gaussian_std(f, states, actions, coupling, transform_action, transform_state)

        # Construct the surrogate loss.
        # Here we still backpropagate through the mean and std, because they are parameterized
        surrogate_loss = grad_mean.detach() * mean
        surrogate_loss += grad_std.detach() * diag_std

        # (N, Da)
        assert surrogate_loss.shape == actions.shape

        # The total derivative is the sum of the partial derivatives wrt each parameter.
        # Sum along the action dimension and average across states.

        surrogate_loss = surrogate_loss.reshape((mc_samples_grad, -1, surrogate_loss.shape[-1])).permute(1, 0, 2)

        if discounts is not None:
            opt_func = (discounts * surrogate_loss.sum(dim=-1).mean(dim=1).reshape((-1, 1))).sum()
        else:
            opt_func = surrogate_loss.sum(dim=-1).mean()

        return opt_func

    def surrogate_opt_sf(self, states_i_t_repeat, actions_i_t_repeat, f, transform_state=lambda x: x,
                         discounts_i_t=None, mc_samples_grad=1):
        """
        Builds the surrogate optimization function to maximize, with gradient computation for score function.
        """
        log_prob_actions_t = self.log_prob_t(states_i_t_repeat.detach(), actions_i_t_repeat.detach())
        states_i_t_repeat = transform_state(states_i_t_repeat)
        states_actions = torch.cat((states_i_t_repeat, actions_i_t_repeat), dim=1)

        try:
            if f.model.__module__.startswith(('sklearn', 'skmultiflow', 'cuml')):
                states_actions = states_actions.detach().cpu().numpy()
        except AttributeError:
            states_actions = states_actions.detach().cpu().numpy()

        f_states_actions = f(states_actions).reshape((states_actions.shape[0], 1))
        if not isinstance(f_states_actions, torch.Tensor):
            f_states_actions = torch.tensor(f_states_actions).to(device=self._torch_device_policy)

        loss = log_prob_actions_t * f_states_actions.detach()
        loss = loss.reshape((mc_samples_grad, -1, 1)).permute(1, 0, 2)

        if discounts_i_t is not None:
            opt_func = (discounts_i_t * loss.mean(dim=1)).sum()
        else:
            opt_func = loss.mean()

        return opt_func

    def surrogate_opt_reptrick(self, states_i_t_repeat, actions_i_t_repeat, f, transform_state=lambda x: x,
                               discounts_i_t=None, mc_samples_grad=1):
        states_i_t_repeat = transform_state(states_i_t_repeat).detach()
        states_actions = torch.cat((states_i_t_repeat, actions_i_t_repeat), dim=1)

        f_states_actions = f(states_actions).reshape((states_actions.shape[0], 1))
        if not isinstance(f_states_actions, torch.Tensor):
            raise NotImplementedError

        loss = f_states_actions
        loss = loss.reshape((mc_samples_grad, -1, 1)).permute(1, 0, 2)

        if discounts_i_t is not None:
            opt_func = (discounts_i_t * loss.mean(dim=1)).sum()
        else:
            opt_func = loss.mean()

        return opt_func

    def generate_mvd_actions(self, states, coupling=False, output_tensors=False):
        actions = self.draw_action_t(states)
        mean, std = self.get_mean_and_std(states, output_tensor=True, match_num_states=True)

        mean_actions_pos, mean_actions_neg, mean_c = self.generate_mvd_theta_mean(states, actions, coupling=coupling)
        std_actions_pos, std_actions_neg, std_c = self.generate_mvd_theta_std(states, actions, coupling=coupling)

        actions_pos_stack = torch.cat((mean_actions_pos, std_actions_pos), dim=0)
        actions_neg_stack = torch.cat((mean_actions_neg, std_actions_neg), dim=0)
        c_stack = torch.cat((mean_c, std_c), dim=0)
        f = detach_and_convert_numpy if not output_tensors else detach_tensors
        actions_pos_stack, actions_neg_stack, c_stack = f(actions_pos_stack, actions_neg_stack, c_stack)

        diff_params = torch.cat((mean, std), dim=0)

        return actions_pos_stack, actions_neg_stack, c_stack, diff_params

    @property
    def mvd_params_dim(self):
        return self._mvd_params_dim


class GaussianTorchPolicyExtendedSquashed(GaussianTorchPolicyExtended):
    """
    Extends to squash actions.
    """

    def __init__(self, network, input_shape, output_shape, min_a, max_a, std_0=1.,
                 trainable_std=True,
                 use_cuda=False, log_std_min=-10, log_std_max=10, **params):

        super().__init__(network, input_shape, output_shape, std_0, trainable_std,
                         use_cuda, log_std_min, log_std_max, **params)

        self._delta_a = to_float_tensor(.5 * (max_a - min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (max_a + min_a), self.use_cuda)

        self._add_save_attr(
            _delta_a='torch',
            _central_a='torch'
        )

    def compute_a_from_u(self, u):
        a = self._central_a + self._delta_a * torch.tanh(u)
        return a

    def compute_u_from_a(self, a):
        u = torch.atanh((a - self._central_a) / self._delta_a)
        return u

    def draw_action_t(self, state, reparametrize=False):
        u = super(GaussianTorchPolicyExtendedSquashed, self).draw_action_t(state, reparametrize)
        return self.compute_a_from_u(u)

    def draw_action(self, state):
        a = self.draw_action_t(state)
        return a.detach().cpu().numpy()

    def mvd_grad_gaussian_mean(self, f, states, actions, coupling=True,
                               transform_action=lambda x: x, transform_state=lambda x: x):
        return super(GaussianTorchPolicyExtendedSquashed, self).mvd_grad_gaussian_mean(
            f, states, actions, coupling, transform_action=self.compute_a_from_u, transform_state=transform_state)

    def mvd_grad_gaussian_std(self, f, states, actions, coupling=True,
                              transform_action=lambda x: x, transform_state=lambda x: x):
        return super(GaussianTorchPolicyExtendedSquashed, self).mvd_grad_gaussian_std(
            f, states, actions, coupling, transform_action=self.compute_a_from_u, transform_state=transform_state)

    def surrogate_opt_mvd(self, states, actions, f, coupling=False,
                          transform_actions=lambda x: x, transform_state=lambda x: x,
                          discounts=None, mc_samples_grad=1):
        return super(GaussianTorchPolicyExtendedSquashed, self).surrogate_opt_mvd(
            states, actions, f, coupling, transform_action=self.compute_a_from_u, transform_state=transform_state,
            discounts=discounts, mc_samples_grad=mc_samples_grad)

    def log_prob_t(self, s, a):
        # a = g(u) = central + delta * tanh(u)
        dist = self.distribution_t(s)
        u = self.compute_u_from_a(a)
        log_prob = dist.log_prob(u)[:, None].sum(dim=1, keepdim=True)
        log_prob -= torch.log(self._delta_a * (1. - torch.tanh(u).pow(2)) + self._eps).sum(dim=1, keepdim=True)
        return log_prob
