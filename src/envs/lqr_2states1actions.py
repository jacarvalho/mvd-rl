import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mushroom_rl.environments.lqr import LQR
from mushroom_rl.solvers.lqr import compute_lqr_V_gaussian_policy, compute_lqr_feedback_gain


class LQR2states1actions(LQR):

    __name__ = 'LQR2states1actions'

    def __init__(self, horizon=1000, gamma=0.99):

        self._sdim = 2
        self._adim = 1

        A = np.array([[-1., 0.2],
                      [0., 0.1]])

        B = np.array([[-0.1],
                      [0.]])

        Q = np.eye(2) * 1.5

        R = np.eye(1) * 1.2

        max_pos, max_action = np.inf, np.inf
        s0 = np.array([[9.] * self._sdim]).reshape((-1,))

        super().__init__(A, B, Q, R,
                         max_pos=max_pos,
                         max_action=max_action,
                         episodic=False,
                         gamma=gamma, horizon=horizon,
                         initial_state=s0,
                         random_init=False,
                         )

    @property
    def obs_space_dim(self):
        return self.info.observation_space.shape[0]

    @property
    def act_space_dim(self):
        return self.info.action_space.shape[0]

    def render(self):
        pass


def compute_stable_K(mdp, initial_std, std=10.):
    s_dim = mdp.info.observation_space.shape[0]
    a_dim = mdp.info.action_space.shape[0]

    K_opt = compute_lqr_feedback_gain(mdp)
    V_opt = compute_lqr_V_gaussian_policy(mdp._initial_state, mdp,
                                          K_opt,
                                          initial_std * np.eye(a_dim))

    eigvals, eigvects = np.linalg.eig(mdp.A)
    assert np.any(np.absolute(eigvals) >= 1), "A matrix must be UNSTABLE"

    print(f'K Optimal: {K_opt}')
    print(f'V_K Optimal: {V_opt}')

    K = np.zeros((a_dim, s_dim))
    while True:
        eigvals, eigvects = np.linalg.eig(mdp.A - mdp.B @ K)
        print(eigvals)
        if np.all(np.absolute(eigvals) < 1):
            break
        K = np.random.multivariate_normal(
                                          K_opt.ravel(),
                                          # np.zeros(s_dim * a_dim),
                                          std*np.eye(s_dim * a_dim))
        K = K.reshape((a_dim, s_dim))

    V_K = compute_lqr_V_gaussian_policy(
        mdp._initial_state, mdp, K, np.atleast_2d(initial_std**2 * np.eye(a_dim)))
    print()
    print(f'K: {K}')
    print(f'V_K: {V_K}')


def plot_stable_region(mdp):
    print("\n----------------------")
    print("Plot STABLE region ")
    assert mdp.info.observation_space.shape[0] == 2, "state space dim must be 2"
    assert mdp.info.action_space.shape[0] == 1, "action space dim must be 1"

    eigvals, eigvects = np.linalg.eig(mdp.A)
    if np.any(np.absolute(eigvals) >= 1):
        print("A matrix is UNSTABLE")
    elif np.all(np.absolute(eigvals) < 1):
        print("A matrix is STABLE")

    K_opt = compute_lqr_feedback_gain(mdp)
    print(f'K Optimal: {K_opt}')

    LIM = 3.
    k1 = np.linspace(-LIM, LIM, num=100)
    k2 = np.linspace(-LIM, LIM, num=100)
    K1, K2 = np.meshgrid(k1, k2)
    k1_flat = K1.flatten().reshape(-1, 1)
    k2_flat = K2.flatten().reshape(-1, 1)
    Z = np.ones_like(K1).ravel()

    for i, (a, b) in enumerate(zip(k1_flat, k2_flat)):
        K = np.stack((a, b), axis=1)
        eigvals, eigvects = np.linalg.eig(mdp.A - mdp.B @ K)
        if np.any(np.absolute(eigvals) >= 1):
            Z[i] = 0

    Z = Z.reshape(K1.shape)

    im = plt.pcolormesh(K1, K2, Z, shading='auto')
    plt.colorbar(im)
    plt.scatter(K_opt.ravel()[0], K_opt.ravel()[1])
    plt.show()


if __name__ == '__main__':
    mdp = LQR2states1actions(horizon=500, gamma=0.99)
    compute_stable_K(mdp, 0.1, std=10.)

    plot_stable_region(mdp)
