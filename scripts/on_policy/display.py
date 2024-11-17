import os

import numpy as np

from matplotlib import cm

from src.utils.dtype_utils import NP_FLOAT_DTYPE

from scripts.plots.basic_settings import *
mpl.rcParams['axes.grid'] = True


def visualization(env_domain, vf, results_dir, n_episodes_test, debug):

    display_func = DisplayNone
    if env_domain == 'Pendulum-v0':
        display_func = DisplayPendulum
    elif env_domain in ['Corridor', 'Room']:
        display_func = DisplayRoom

    visualization_callback = display_func(vf,
                                          results_dir,
                                          n_episodes_test,
                                          debug)

    return visualization_callback


class ValueFunction:

    def __init__(self, agent):
        self._agent = agent

    def __call__(self, s):
        s_repeat = np.repeat(s, 10, axis=0)
        a_repeat = self._agent.policy.draw_action(s_repeat)
        sa = np.concatenate((s_repeat, a_repeat), axis=1)
        v = np.mean(self._agent._Q_approx(sa).reshape(s.shape[0], -1), axis=1)
        return v


class Display2D:
    def __init__(self, vf, phi, psi, process_dataset, low, high, results_dir,
                 n_episodes_test, labels, debug, plot_trajectories=True):
        if debug:
            plt.ion()

        self._V = vf
        self._phi = phi
        self._psi = psi
        self._process_dataset = process_dataset

        self._results_dir = results_dir

        self._plot_trajectories = plot_trajectories

        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)

        self._s1 = np.linspace(low[0], high[0], 50, dtype=NP_FLOAT_DTYPE)
        self._s2 = np.linspace(low[1], high[1], 50, dtype=NP_FLOAT_DTYPE)

        try:
            vv, mm, x, y = self._compute_data(None)
        except:
            vv, _ = np.meshgrid(self._s1, self._s2)
            mm = vv

        ext = [low[0], high[0],
               low[1], high[1]]

        # ax1.set_title('$V^{\pi}$', y=1.10)
        # ax1.set_xlabel(f'${labels[0]}$')
        # ax1.set_ylabel(f'${labels[1]}$', rotation=0, labelpad=13)
        ax1.set_xticks([])
        ax1.set_yticks([])

        self._ax1 = ax1
        im1 = ax1.imshow(vv, cmap=cm.coolwarm, extent=ext, aspect='auto', origin='lower')
        self._sc = []
        self._first_points = []
        self._last_points = []
        for _ in range(n_episodes_test):
            sc, = ax1.plot([], [], color='black', alpha=0.8, linewidth=0.5)
            self._sc += [sc]
            fp = ax1.scatter([], [], color='black', marker='D', alpha=0.8, s=25.0)
            lp = ax1.scatter([], [], color='black', marker='o', alpha=1.0, s=25.0)
            self._first_points += [fp]
            self._last_points += [lp]

        self._im = [im1]

        self._counter = 0

        plt.draw()
        plt.tight_layout()
        plt.savefig(os.path.join(self._results_dir, 'valuefunction-' + str(0) + '.pdf'), dpi=300)

    def __call__(self, dataset, epoch, *args, **kwargs):
        try:
            vv, x, y = self._compute_data(dataset)
        except:
            vv, _ = np.meshgrid(self._s1, self._s2)
            x, y = self._process_dataset(dataset)

        # self._ax1.set_title(fr'$V^{{\pi}}$ - Epoch {str(epoch)}')
        self._im[0].set_data(vv)
        self._im[0].autoscale()
        if self._plot_trajectories:
            for i, (xs, ys) in enumerate(zip(x, y)):
                self._sc[i].set_data(xs, ys)
                self._first_points[i].set_offsets((xs[0], ys[0]))
                self._last_points[i].set_offsets((xs[-1], ys[-1]))
        # else:
        #     self._last_points[0].set_offsets(np.stack((np.asarray(x).ravel(), np.asarray(y).ravel())).T)

        self._counter += 1

        plt.draw()
        plt.tight_layout()
        plt.savefig(os.path.join(self._results_dir, 'valuefunction-' + str(epoch) + '.pdf'), dpi=300)

    def _compute_data(self, dataset):
        if dataset is None:
            x, y = [], []
        else:
            x, y = self._process_dataset(dataset)

        s1 = self._s1
        s2 = self._s2

        X, Y = np.meshgrid(s1, s2)
        X_flat = self._phi(X.flatten().reshape(-1, 1))
        Y_flat = self._psi(Y.flatten().reshape(-1, 1))
        XY_flat = np.hstack((X_flat, Y_flat))

        vv = self._V(XY_flat).reshape(X.shape)

        return vv, x, y


class DisplayPendulum(Display2D):
    def __init__(self, vf, results_dir, n_episodes_test, debug):

        phi = lambda x: np.hstack((np.cos(x), np.sin(x)))
        psi = lambda x: x
        low = [-np.pi, -8.]
        high = [np.pi, 8.]

        labels = [r'\theta', r'\dot{\theta}']

        super(DisplayPendulum, self).__init__(vf, phi, psi, self.process_dataset, low, high, results_dir,
                                              n_episodes_test, labels, debug, plot_trajectories=False)

    @staticmethod
    def process_dataset(dataset):
        x_l = []
        y_l = []
        x = []
        y = []
        for transition in dataset.get():
            state = np.asarray(transition[0], dtype=NP_FLOAT_DTYPE)
            _x = np.arctan2(state[1], state[0])
            _y = state[2]

            x.append(_x)
            y.append(_y)

            if transition[-1]:
                x_l.append(x.copy())
                y_l.append(y.copy())
                x = []
                y = []

        return x_l, y_l


class DisplayRoom(Display2D):
    def __init__(self, vf, results_dir, n_episodes_test, debug):
        phi = lambda x: x
        psi = phi
        low = [-1., -1.]
        high = [1., 1.]

        labels = [r'x', r'y']

        super(DisplayRoom, self).__init__(vf, phi, psi, self.process_dataset, low, high, results_dir,
                                          n_episodes_test, labels, debug, plot_trajectories=True)

    @staticmethod
    def process_dataset(dataset):
        x_l = []
        y_l = []
        x = []
        y = []
        for transition in dataset.get():
            x.append(transition[0][0])
            y.append(transition[0][1])

            if transition[-1]:
                x.append(transition[3][0])
                y.append(transition[3][1])
                x_l.append(x.copy())
                y_l.append(y.copy())
                x = []
                y = []

        return x_l, y_l


class DisplayNone(Display2D):

    def __init__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        pass
