import numpy as np

from mushroom_rl.algorithms.policy_search.black_box_optimization import BlackBoxOptimization


class PGPE(BlackBoxOptimization):
    """
    Policy Gradient with Parameter Exploration algorithm.
    "A Survey on Policy Search for Robotics", Deisenroth M. P., Neumann G.,
    Peters J.. 2013.

    """
    def __init__(self, mdp_info, distribution, policy, grad_optimizer,
                 mc_samples_gradient=1,
                 baseline=True,
                 features=None):
        """
        Constructor.

        Args:
            grad_optimizer: the gradient optimizer

        """
        self.mc_samples_gradient = mc_samples_gradient
        self.grad_optimizer = grad_optimizer
        self._baseline = baseline

        self._add_save_attr(learning_rate='pickle')

        super().__init__(mdp_info, distribution, policy, features)

    def _update(self, Jep, theta):
        grad_J_list = self.get_gradients(self.distribution, Jep, theta, self._baseline)

        grad_J = np.mean(grad_J_list, axis=0)

        omega = self.distribution.get_parameters()
        omega = self.grad_optimizer(omega, grad_J)
        self.distribution.set_parameters(omega)

    @staticmethod
    def get_gradients(distribution, Jep, theta, _baseline=False):
        baseline_num_list = list()
        baseline_den_list = list()
        diff_log_dist_list = list()

        # Compute derivatives of distribution and baseline components
        for i in range(len(Jep)):
            J_i = Jep[i]
            theta_i = theta[i]

            diff_log_dist = distribution.diff_log(theta_i)
            diff_log_dist2 = diff_log_dist**2

            diff_log_dist_list.append(diff_log_dist)
            baseline_num_list.append(J_i * diff_log_dist2)
            baseline_den_list.append(diff_log_dist2)

        # Compute baseline
        baseline = np.mean(baseline_num_list, axis=0) / \
            np.mean(baseline_den_list, axis=0)
        baseline[np.logical_not(np.isfinite(baseline))] = 0.

        # Compute gradient
        grad_J_list = list()
        for i in range(len(Jep)):
            diff_log_dist = diff_log_dist_list[i]
            J_i = Jep[i]

            grad_J_list.append(diff_log_dist * (J_i - (baseline if _baseline else 0.)))

        return grad_J_list
