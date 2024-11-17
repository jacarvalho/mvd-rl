import numpy as np

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization
from mushroom_rl.utils.numerical_gradient import numerical_diff_function

from src.distributions.gaussians import GaussianDistributionDiagonalCovarianceLogParameterization


class RepTrick(BlackBoxOptimization):
    """
    Reparametrization trick for Black Box Optimization.

    """
    def __init__(self, mdp_info, distribution, policy, function, grad_optimizer,
                 mc_samples_gradient=1,
                 features=None):
        """
        Constructor.

        Args:
            grad_optimizer: the gradient optimizer

        """
        assert isinstance(distribution, GaussianDistributionDiagonalCovarianceLogParameterization), "distribution not supported"

        self._function = function

        self.mc_samples_gradient = mc_samples_gradient
        self.grad_optimizer = grad_optimizer

        self._add_save_attr(learning_rate='pickle')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        grad_J_list = self.get_gradients(theta)

        grad_J = np.mean(grad_J_list, axis=0)

        omega = self.distribution.get_parameters()
        omega = self.grad_optimizer(omega, grad_J)
        self.distribution.set_parameters(omega)

    def get_gradients(self, theta):
        grad_J_list = []
        for th in theta:
            grad_dist_parameters = self.distribution.diff_reptrick(th)

            # grad of the function with finite differences
            grad_function = numerical_diff_function(self._function, th, eps=1e-6)

            n = len(grad_dist_parameters) // len(grad_function)
            grad_function_cat = np.concatenate([grad_function for _ in range(n)])

            grad_J_list.append(grad_dist_parameters * grad_function_cat)

        return grad_J_list
