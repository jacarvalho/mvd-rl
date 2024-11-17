import os

from src.distributions.gaussians import GaussianDistributionFixedFullCovariance
from src.envs.optim_test_functions import *

from scripts.plots.basic_settings import *
mpl.rcParams['text.usetex'] = True


########################################################################################################################
experiment_dir_base = '../../logs/test-functions'

algs = {
        'EMVD': {'name': 'MVD',
                 'color': 'blue',
                 'zorder': 3},
        'RepTrick': {'name': 'RepTrick',
                     'color': 'orange',
                     'zorder': 2},
        'PGPE': {'name': 'SF',
                 'color': 'green',
                 'zorder': 1},
        }

functions = {
             'Quadratic': {'func': Quadratic(), 'lims_x': [-5.5, 0.05], 'lims_y': [-5.5, 0.05]},
             'Himmelblau': {'func': Himmelblau(), 'lims_x': [-4.3, 4.0], 'lims_y': [-6.3, 2.]},
             'Styblinski': {'func': Styblinski(), 'lims_x': [-3.2, 0.25], 'lims_y': [-3.2, 0.27]}
             }


for i, function in enumerate(functions):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    # Function contour
    granularity = 250
    func = functions[function]['func']
    lim_x = functions[function]['lims_x']
    lim_y = functions[function]['lims_y']
    _x1 = np.linspace(lim_x[0], lim_x[1], num=granularity, dtype=np.float32)
    _x2 = np.linspace(lim_y[0], lim_y[1], num=granularity, dtype=np.float32)
    X1, X2 = np.meshgrid(_x1, _x2)
    X1_flat, X2_flat = X1.flatten(), X2.flatten()
    Z = np.zeros_like(X1_flat, dtype=np.float32)
    for j, (x1, x2) in enumerate(zip(X1_flat, X2_flat)):
        x = np.array([[x1, x2]], dtype=np.float32)
        _, reward, _, _ = func.step(x)
        Z[j] = reward
    Z = Z.reshape(X1.shape)

    im = axs.contour(X1, X2, Z, levels=50, cmap=mpl.cm.RdGy, alpha=0.3, zorder=-2)
    axs.set(aspect='equal')
    axs.set_xticks([])
    axs.set_yticks([])
    axs.set_xlim(lim_x[0], lim_x[1])
    axs.set_ylim(lim_y[0], lim_y[1])

    for alg in algs:
        means = np.load(os.path.join(experiment_dir_base, function, alg, 'means.npy'))
        covs = np.load(os.path.join(experiment_dir_base, function, alg, 'covs.npy'))

        # 2D plot
        total_epochs = means.shape[0] - 2
        for k in [0, total_epochs]:
            if covs[k].ndim == 1:
                cov = np.diag(covs[k])
            else:
                cov = covs[k]
            dist = GaussianDistributionFixedFullCovariance(means[k], cov)
            z = dist.sample(size=1000)
            confidence_ellipse(z[:, 0], z[:, 1], axs, n_std=1.0, edgecolor=algs[alg]['color'], linewidth=3.)

        line, = axs.plot(means[:, 0], means[:, 1], color=algs[alg]['color'], linewidth=3., zorder=algs[alg]['zorder'])
        axs.scatter(means[0, 0], means[0, 1], color='red', marker='D', linewidths=1, zorder=4, s=300)
        axs.scatter(means[total_epochs, 0], means[total_epochs, 1], color=algs[alg]['color'], marker='o',
                    linewidths=1,
                    zorder=algs[alg]['zorder'], s=300)

    fig.tight_layout()
    fig.savefig(f'./out/test_function_{function}.pdf', bbox_inches='tight', dpi=300)

# Legends
for alg in algs:
    axs.plot([], [], c=algs[alg]['color'], label=algs[alg]['name'])
fig.legend(bbox_to_anchor=(0.5, 0.1), loc="center", bbox_transform=fig.transFigure, ncol=3, frameon=False)

export_legend(axs, filename="out/legend_test-functions.pdf")
