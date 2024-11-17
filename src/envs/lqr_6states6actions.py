import numpy as np

from mushroom_rl.environments.lqr import LQR
from mushroom_rl.solvers.lqr import compute_lqr_V_gaussian_policy, compute_lqr_feedback_gain

from src.envs.lqr_2states1actions import compute_stable_K


class LQR6states6actions(LQR):

    __name__ = 'LQR6states6actions'

    def __init__(self, horizon=1000, gamma=0.99):

        self._sdim = 6
        self._adim = 6

        A = np.array([[-0.1, -0.5, 0., 0., 0.3, 0.1],
                      [0., 0.1, 2., 0., -0.1, 0.2],
                      [-2., 0., 0.1, 0., -0.4, 0.8],
                      [0., -0.2, 0., 0.1, 0.2, 0.2],
                      [0., -0.1, 0.2, 0.1, -0.1, 0.1],
                      [0.1, 0.1, 0.1, 0., 0., 0.2],
                      ])

        B = np.array([[-1., 0, 0.2, 0.5, 0., 0.],
                      [-1., 0.1, 0., 0., 0., 0.1],
                      [0., -1., 0.1, 0., 0., 0.],
                      [0., 0.1, 0., 0.1, -0.2, 0.3],
                      [0.1, 0.2, -0.1, 0., -0.1, 0.],
                      [0., 0.1, 0.1, 0.1, 1.2, 0.1],
                      ])

        Q = np.eye(self._sdim) * 1.5

        R = np.eye(self._adim) * 1.2

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


if __name__ == '__main__':
    mdp = LQR6states6actions(horizon=500, gamma=0.99)
    compute_stable_K(mdp, 0.1, std=1.)
