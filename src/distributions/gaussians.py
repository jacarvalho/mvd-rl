import numpy as np
from scipy.stats import multivariate_normal, weibull_min

from mushroom_rl.distributions.distribution import Distribution

from src.distributions.torchdist_utils import DoubleSidedStandardMaxwell, standard_gaussian_from_standard_dsmaxwell


def conditionalMultivariateGaussian(rs, mean, cov, idx):
    """
    Computes the mean and covariance of the multivariate Gaussian
    of the idx dimension conditioned on all other dimensions.

    Args:
        rs: sample from joint Gaussian
        mean: mean of the joint Gaussian
        cov: covariance of the joint Gaussian
        idx: index to return the conditional Gaussian(idx|all indexes expect idx)

    Returns:
        mean and covariance of conditioned Gaussian

    """
    if rs.ndim == 2:
        rs = rs.reshape(-1,)
    assert 0 <= idx < len(rs), "idx is out of bounds"
    # Swap indices of idx and 0
    # Swap indices of the random sample
    rs[[0, idx]] = rs[[idx, 0]]
    # Swap indices of mean
    mean[[0, idx]] = mean[[idx, 0]]
    # Swap columns and then rows of covariance
    cov[:, [0, idx]] = cov[:, [idx, 0]]
    cov[[0, idx], :] = cov[[idx, 0], :]

    # Conditioned mean and cov
    x2 = np.delete(rs, 0)
    mu2 = np.delete(mean, 0)
    cov11 = cov[0, 0]
    cov22 = np.delete(np.delete(cov, 0, axis=0), 0, axis=1)
    cov12 = cov[0, 1:]
    cov21 = cov[1:, 0]

    mean_x1_given_x2 = mean[0] + cov12 @ np.linalg.inv(cov22) @ (x2 - mu2)
    cov_x1_given_x2 = cov11 - cov12 @ np.linalg.inv(cov22) @ cov21

    return mean_x1_given_x2, cov_x1_given_x2


class GaussianDistributionFixedFullCovariance(Distribution):
    """
    Gaussian distribution with fixed full covariance matrix. The parameters
    vector represents only the mean.

    """
    def __init__(self, mu, sigma):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            sigma (np.ndarray): covariance matrix of the distribution.

        """
        self._dim = mu.shape[0]

        self._mu = mu
        self._sigma = sigma
        self._inv_sigma = np.linalg.inv(sigma)

        self._std_conditioned_sampled = None

        self._add_save_attr(
            _mu='numpy',
            _sigma='numpy',
            _inv_sigma='numpy'
        )

    def sample(self, size=1):
        return np.random.multivariate_normal(self._mu, self._sigma, size=size)

    def log_pdf(self, theta):
        return multivariate_normal.logpdf(theta, self._mu, self._sigma)

    def __call__(self, theta):
        return multivariate_normal.pdf(theta, self._mu, self._sigma)

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(2*np.pi*np.e*self._sigma))

    def mle(self, theta, weights=None):
        if weights is None:
            self._mu = np.mean(theta, axis=0)
        else:
            self._mu = weights.dot(theta) / np.sum(weights)

    def diff_log(self, theta):
        delta = (theta - self._mu).reshape(-1,)
        g = self._inv_sigma @ delta

        return g

    def get_parameters(self):
        return self._mu

    def set_parameters(self, rho):
        self._mu = np.copy(rho)

    def print_dist_info(self):
        with np.printoptions(precision=3):
            print('mean, cov: ', self.mean, self.cov)

    @property
    def dim(self):
        return self._dim

    @property
    def mean(self):
        return np.copy(self._mu)

    @property
    def cov(self):
        return np.copy(self._sigma)

    @property
    def parameters_size(self):
        return len(self._mu)

    @property
    def theta_mvd_size(self):
        return 2 * self.parameters_size

    @property
    def theta_fd_size(self):
        return 2 * self.parameters_size

    @property
    def params_per_dim(self):
        # the mean
        return 1

    def reset_std_conditioned(self):
        self._std_conditioned_sampled = None

    def generate_mvd_theta(self, coupling=False, grad_idxs=None, grad_params_idxs=None):
        if grad_params_idxs is not None:
            grad_idxs = grad_params_idxs
        theta_mean = self.generate_mvd_theta_mean(coupling, grad_idxs)
        return theta_mean

    def generate_mvd_theta_mean(self, coupling=False, grad_idxs=None):
        """
        Generate a set of thetas according to the mvd of the mean.
        """
        mean, cov = self.mean, self.cov

        self.reset_std_conditioned()

        dist_sample = np.atleast_2d(self.sample())

        # Replicate the last dimension
        # (D, D)
        base_sample = dist_sample
        multiples = [self._dim, 1]
        base_sample = np.tile(base_sample, multiples)

        # Sample (D) samples from the positive and negative Univariate Weibull distributions
        pos_sample_weibull = weibull_min.rvs(2, loc=0, scale=np.sqrt(2), size=dist_sample.shape)

        if coupling:
            neg_sample_weibull = pos_sample_weibull
        else:
            neg_sample_weibull = weibull_min.rvs(2, loc=0, scale=np.sqrt(2), size=dist_sample.shape)

        # Build the (D) positive and negative diagonals of the MVD decomposition
        # Construct the conditioned means and conditioned covariances
        mean_conditioned = np.zeros(self._dim)
        cov_conditioned = np.zeros(self._dim)  # the covariance is one-dimensional
        for i in range(self._dim):
            mean_conditioned[i], cov_conditioned[i] = conditionalMultivariateGaussian(dist_sample, mean, cov, i)

        std_conditioned = np.sqrt(cov_conditioned)
        self._std_conditioned_sampled = std_conditioned

        pos_diag = mean_conditioned + std_conditioned * pos_sample_weibull
        assert pos_diag.shape == dist_sample.shape

        neg_diag = mean_conditioned - std_conditioned * neg_sample_weibull
        assert neg_diag.shape == dist_sample.shape

        # Set the positive and negative thetas.
        # (D, D)
        # Replace the ith dimension with the ith entry of the constructed diagonals.
        pos_theta = base_sample.copy()
        np.fill_diagonal(pos_theta, pos_diag)
        neg_theta = base_sample.copy()
        np.fill_diagonal(neg_theta, neg_diag)

        theta = np.vstack((pos_theta, neg_theta))

        if grad_idxs is not None:
            theta = np.vstack((theta[grad_idxs], theta[grad_idxs + self._mu.size]))

        return theta

    def mvd_omega(self, theta, f_theta, grad_idxs=None):
        """
        MVD estimate for mean for one MC estimate.
        """
        if grad_idxs is not None:
            f_theta_aux = np.zeros((self.theta_mvd_size, 1))
            grad_idxs_aux = np.hstack((grad_idxs, grad_idxs + self.dim))
            f_theta_aux[grad_idxs_aux] = f_theta
            f_theta = f_theta_aux

        idx_theta_mean = 2 * self._dim
        f_theta_mean = f_theta[:idx_theta_mean]

        # Mean
        c_mean = np.sqrt(2 * np.pi) * self._std_conditioned_sampled.reshape(-1, 1)
        delta_f_mean = f_theta_mean[:self._dim] - f_theta_mean[self._dim:]

        grad_mean = delta_f_mean / c_mean
        grad = grad_mean
        return grad

    def grad_mvd_params(self, theta, f_theta, grad_idxs=None, grad_params_idxs=None):
        if grad_params_idxs is not None:
            grad_idxs = grad_params_idxs
        grad = self.mvd_omega(theta, f_theta, grad_idxs=grad_idxs)
        return grad

    def generate_fd_theta(self, h=1e-3):
        mean, cov = self.mean, self.cov

        # Mean
        id_matrix = np.identity(self._dim)
        means_pos = mean + h * id_matrix
        means_neg = mean - h * id_matrix
        means = np.vstack((means_pos, means_neg))
        theta_means = np.array([np.random.multivariate_normal(m, cov) for m in means])

        theta = theta_means
        return theta

    def fd_omega(self, theta, f_theta, h=1e-3):
        """
        FD estimate for mean and std for one MC estimate.
        """
        idx_theta_mean = 2 * self._dim
        f_theta_mean = f_theta[:idx_theta_mean]

        # Mean
        delta_f_mean = f_theta_mean[:self._dim] - f_theta_mean[self._dim:]
        grad_mean = delta_f_mean / 2*h

        grad = grad_mean
        return grad

    def grad_fd_params(self, theta, f_theta, h=1e-3):
        return self.fd_omega(theta, f_theta, h)


class GaussianDistributionDiagonalCovariance(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix.
    The parameters vector represents the mean and the standard deviation for each dimension.

    """
    def __init__(self, mu, std):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            std (np.ndarray): initial vector of standard deviations for each
                variable of the distribution.

        """
        assert(len(std.shape) == 1)
        self._dim = mu.shape[0]

        self._mu = mu
        assert np.all(std) > 0, "Standard deviation must be greater than 0."
        self._std = std

        self._add_save_attr(
            _mu='numpy',
            _std='numpy'
        )

    def sample(self, size=None):
        sigma = np.diag(self._std**2)
        return np.random.multivariate_normal(self._mu, sigma, size=size)

    def log_pdf(self, theta):
        sigma = np.diag(self._std ** 2)
        return multivariate_normal.logpdf(theta, self._mu, sigma)

    def __call__(self, theta):
        sigma = np.diag(self._std ** 2)
        return multivariate_normal.pdf(theta, self._mu, sigma)

    def entropy(self):
        return 0.5 * np.log(np.product(2*np.pi*np.e*self._std**2))

    def mle(self, theta, weights=None):
        if weights is None:
            self._mu = np.mean(theta, axis=0)
            self._std = np.std(theta, axis=0)
        else:
            sumD = np.sum(weights)
            sumD2 = np.sum(weights**2)
            Z = sumD - sumD2 / sumD

            self._mu = weights.dot(theta) / sumD

            delta2 = (theta - self._mu)**2
            self._std = np.sqrt(weights.dot(delta2) / Z)

    def diff_log(self, theta):
        n_dims = len(self._mu)

        sigma = self._std**2

        g = np.empty(self.parameters_size)

        delta = theta - self._mu

        g_mean = delta / sigma
        g_std = delta**2 / (self._std**3) - 1 / self._std

        g[:n_dims] = g_mean
        g[n_dims:] = g_std
        return g

    def get_parameters(self):
        rho = np.empty(self.parameters_size)
        n_dims = len(self._mu)

        rho[:n_dims] = self._mu
        rho[n_dims:] = self._std

        return rho

    def set_parameters(self, rho):
        n_dims = len(self._mu)
        self._mu = np.copy(rho[:n_dims])
        self._std = np.copy(rho[n_dims:])

    def print_dist_info(self):
        with np.printoptions(precision=3):
            print('mean, cov: ', self.mean, self.cov)

    @property
    def mean(self):
        return np.copy(self._mu)

    @property
    def cov(self):
        return np.copy(self._std**2)

    @property
    def dim(self):
        return self._dim

    @property
    def parameters_size(self):
        return 2 * len(self._mu)

    @property
    def theta_mvd_size(self):
        return 2 * self.parameters_size

    @property
    def theta_fd_size(self):
        return 2 * self.parameters_size

    @property
    def params_per_dim(self):
        # the mean and std
        return 2

    def generate_mvd_theta(self, coupling=False, grad_dim_idxs=None, grad_params_idxs=None):
        grad_idxs_mean = grad_dim_idxs
        grad_idxs_std = grad_dim_idxs
        if grad_params_idxs is not None:
            grad_idxs_mean = grad_params_idxs[grad_params_idxs < self._dim]
            grad_idxs_std = grad_params_idxs[grad_params_idxs >= self._dim] - self._dim
        theta_mean = self.generate_mvd_theta_mean(coupling, grad_idxs_mean)
        theta_std = self.generate_mvd_theta_std(coupling, grad_idxs_std)
        theta = np.vstack((theta_mean, theta_std))
        return theta

    def generate_mvd_theta_mean(self, coupling=False, grad_idxs=None):
        """
        Generate a set of thetas according to the mvd of the mean.
        """
        mean, std = self._mu, self._std

        dist_sample = np.atleast_2d(self.sample())

        # Replicate the last dimension
        # (D, D)
        base_sample = dist_sample
        multiples = [self._dim, 1]
        base_sample = np.tile(base_sample, multiples)

        # Sample (D) samples from the positive and negative Univariate Weibull distributions
        pos_sample_weibull = weibull_min.rvs(2, loc=0, scale=np.sqrt(2), size=dist_sample.shape)

        if coupling:
            neg_sample_weibull = pos_sample_weibull
        else:
            neg_sample_weibull = weibull_min.rvs(2, loc=0, scale=np.sqrt(2), size=dist_sample.shape)

        # Build the (D) positive and negative diagonals of the MVD decomposition
        pos_diag = mean + std * pos_sample_weibull
        assert pos_diag.shape == dist_sample.shape

        neg_diag = mean - std * neg_sample_weibull
        assert neg_diag.shape == dist_sample.shape

        # Set the positive and negative thetas.
        # (D, D)
        # Replace the ith dimension with the ith entry of the constructed diagonals.
        pos_theta = base_sample.copy()
        np.fill_diagonal(pos_theta, pos_diag)
        neg_theta = base_sample.copy()
        np.fill_diagonal(neg_theta, neg_diag)

        theta = np.vstack((pos_theta, neg_theta))

        if grad_idxs is not None:
            theta = np.vstack((theta[grad_idxs], theta[grad_idxs + self._mu.size]))

        return theta

    def generate_mvd_theta_std(self, coupling=False, grad_idxs=None):
        """
        Generate a set of thetas according to the mvd of the std.
        """
        mean, std = self._mu, self._std

        dist_sample = np.atleast_2d(self.sample())

        # Replicate the last dimension
        # (D, D)
        base_sample = dist_sample
        multiples = [self._dim, 1]
        base_sample = np.tile(base_sample, multiples)

        # Sample (D) samples from the positive and negative components
        # The positive part is a Double-sided Maxwell M(mu, sigma^2).
        #   To sample Y from the Double-sided Maxwell M(mu, sigma^2) we can do
        #   X ~ M(0, 1) -> Y = mu + sigma * X
        # The negative part is a Gaussian distribution N(mu, sigma^2).
        #   To sample Y from the Gaussian N(mu, sigma^2) we can do
        #   X ~ N(0, 1) -> Y = mu + sigma * X
        double_sided_maxwell_standard = DoubleSidedStandardMaxwell()
        pos_sample_double_sided_maxwell_standard = double_sided_maxwell_standard.sample(dist_sample.shape).cpu().detach().numpy()
        pos_sample_double_sided_maxwell_standard = np.atleast_2d(pos_sample_double_sided_maxwell_standard)

        if coupling:
            # Construct standard Gaussian samples from standard Double-sided Maxwell samples
            neg_sample_gaussian_standard = standard_gaussian_from_standard_dsmaxwell(
                pos_sample_double_sided_maxwell_standard).cpu().detach().numpy()
        else:
            neg_sample_gaussian_standard = np.random.normal(size=dist_sample.shape)

        # Build the (D) positive and negative diagonals of the MVD decomposition
        pos_diag = mean + std * pos_sample_double_sided_maxwell_standard
        assert pos_diag.shape == dist_sample.shape

        neg_diag = mean - std * neg_sample_gaussian_standard
        assert neg_diag.shape == dist_sample.shape

        # Set the positive and negative thetas.
        # (D, D)
        # Replace the ith dimension with the ith entry of the constructed diagonals.
        pos_theta = base_sample.copy()
        np.fill_diagonal(pos_theta, pos_diag)
        neg_theta = base_sample.copy()
        np.fill_diagonal(neg_theta, neg_diag)

        theta = np.vstack((pos_theta, neg_theta))

        if grad_idxs is not None:
            theta = np.vstack((theta[grad_idxs], theta[grad_idxs + self._std.size]))

        return theta

    def mvd_omega(self, theta, f_theta, grad_dims_idxs=None, grad_params_idxs=None):
        """
        MVD estimate for mean and std for one MC estimate.
        """
        mean, std = self._mu.reshape(-1, 1), self._std.reshape(-1, 1)

        if grad_dims_idxs is not None or grad_params_idxs is not None:
            f_theta_aux = np.zeros((self.theta_mvd_size, 1))
            idx_mean, idx_std = None, None
            if grad_dims_idxs is not None:
                idx_mean = np.hstack((grad_dims_idxs, grad_dims_idxs + self.dim))
                idx_std = idx_mean + 2 * self._mu.size
            elif grad_params_idxs is not None:
                grad_idxs_mean = grad_params_idxs[grad_params_idxs < self._dim]
                grad_idxs_std = grad_params_idxs[grad_params_idxs >= self._dim] - self._dim
                idx_mean = np.hstack((grad_idxs_mean, grad_idxs_mean + self.dim))
                idx_std = np.hstack((grad_idxs_std + 2 * self.dim, grad_idxs_std + 3 * self.dim))
            grad_idxs_aux = np.hstack((idx_mean, idx_std))
            f_theta_aux[grad_idxs_aux] = f_theta
            f_theta = f_theta_aux

        idx_theta_mean = 2 * self._dim
        theta_mean = theta[:idx_theta_mean]
        f_theta_mean = f_theta[:idx_theta_mean]

        theta_std = theta[idx_theta_mean:]
        f_theta_std = f_theta[idx_theta_mean:]

        # Mean
        c_mean = 1 / (np.sqrt(2 * np.pi) * std)
        delta_f_mean = f_theta_mean[:self._dim] - f_theta_mean[self._dim:]
        grad_mean = c_mean * delta_f_mean

        # Std
        c_std = 1 / std
        delta_f_std = f_theta_std[:self._dim] - f_theta_std[self._dim:]
        grad_std = c_std * delta_f_std

        grad = np.vstack((grad_mean, grad_std))
        return grad

    def grad_mvd_params(self, theta, f_theta, grad_dims_idxs=None, grad_params_idxs=None):
        return self.mvd_omega(theta, f_theta, grad_dims_idxs, grad_params_idxs)

    def generate_fd_theta(self, h=1e-3):
        mean, std = self._mu, self._std

        # Mean
        id_matrix = np.identity(self._dim)
        means_pos = mean + h * id_matrix
        means_neg = mean - h * id_matrix
        means = np.vstack((means_pos, means_neg))
        sigma = np.diag(self._std**2)
        theta_means = np.array([np.random.multivariate_normal(m, sigma) for m in means])

        # Std
        id_matrix = np.identity(self._dim)
        stds_pos = std + h * id_matrix
        stds_neg = std - h * id_matrix
        stds = np.vstack((stds_pos, stds_neg))
        theta_stds = np.array([np.random.multivariate_normal(mean, np.diag(s**2)) for s in stds])

        theta = np.vstack((theta_means, theta_stds))
        return theta

    def fd_omega(self, theta, f_theta, h=1e-3):
        """
        FD estimate for mean and std for one MC estimate.
        """
        idx_theta_mean = 2 * self._dim
        f_theta_mean = f_theta[:idx_theta_mean]
        f_theta_std = f_theta[idx_theta_mean:]

        # Mean
        delta_f_mean = f_theta_mean[:self._dim] - f_theta_mean[self._dim:]
        grad_mean = delta_f_mean / 2*h

        # Std
        delta_f_std = f_theta_std[:self._dim] - f_theta_std[self._dim:]
        grad_std = delta_f_std / 2*h

        grad = np.vstack((grad_mean, grad_std))
        return grad

    def grad_fd_params(self, theta, f_theta, h=1e-3):
        return self.fd_omega(theta, f_theta, h)


class GaussianDistributionDiagonalCovarianceLogParameterization(GaussianDistributionDiagonalCovariance):
    """
    Gaussian distribution with diagonal covariance matrix.
    The parameters vector represents the mean and the log standard deviation for each dimension.

    """
    def __init__(self, mu, std):
        """
        Constructor.

        Args:
            mu (np.ndarray): initial mean of the distribution;
            std (np.ndarray): initial vector of standard deviations for each variable of the distribution.

        """
        super(GaussianDistributionDiagonalCovarianceLogParameterization, self).__init__(mu, std)
        self._log_std = np.log(std)

    def diff_log(self, theta):
        n_dims = len(self._mu)

        sigma = self._std**2

        g = np.empty(self.parameters_size)

        delta = theta - self._mu

        g_mean = delta / sigma
        g_std = delta**2 / (self._std**3) - 1 / self._std
        g_log_std = g_std * np.exp(self._log_std)

        g[:n_dims] = g_mean
        g[n_dims:] = g_log_std
        return g

    def diff_reptrick(self, theta):
        n_dims = len(self._mu)
        g = np.empty(self.parameters_size)
        g[:n_dims] = np.ones(n_dims)
        g_std = 1 / self._std * (theta - self._mu)
        g_log_std = np.exp(self._log_std) * g_std
        g[n_dims:] = g_log_std

        return g

    def get_parameters(self):
        rho = np.empty(self.parameters_size)
        rho[:self._dim] = self._mu
        rho[self._dim:] = self._log_std
        return rho

    def set_parameters(self, rho):
        self._mu = np.copy(rho[:self._dim])
        self._log_std = np.copy(rho[self._dim:])
        self._std = np.exp(self._log_std)

    @property
    def cov(self):
        return np.copy(np.exp(2 * self._log_std))

    def grad_mvd_params(self, theta, f_theta, grad_dims_idxs=None, grad_params_idxs=None):
        mvd_omega = self.mvd_omega(theta, f_theta, grad_dims_idxs, grad_params_idxs)
        mvd_omega[self._dim:] = mvd_omega[self._dim:] * np.exp(self._log_std).reshape((-1, 1))
        return mvd_omega

    def grad_fd_params(self, theta, f_theta, h=1e-3):
        fd_omega = self.fd_omega(theta, f_theta)
        fd_omega[self._dim:] = fd_omega[self._dim:] * np.exp(self._log_std).reshape((-1, 1))
        return self.fd_omega(theta, f_theta, h)
