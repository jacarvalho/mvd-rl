import json
import os
from scipy.spatial.distance import cosine

from basic_settings import *

mpl.rcParams['lines.linewidth'] = 1.
mpl.rcParams['text.usetex'] = True
plt.rcParams["axes.grid.axis"] = "both"
plt.rcParams["axes.grid"] = True
ALPHA = 0.35

########################################################################################################################
experiment_dir_base = f'../../logs/lqr_pg'

env_domain_d = {'LQR2states1actions': None,
                'LQR2states2actions': None,
                'LQR4states4actions': None,
                'LQR6states6actions': None
                }

N_EPISODES_l = [1, 5, 10, 25]

algs = {
        'mvd': {'name': 'MVD',
                'color': 'blue',
                'linestyle': '-',
                'zorder': 3
                },
        'reptrick': {'name': 'RepTrick',
                     'color': 'orange',
                     'linestyle': '-',
                     'zorder': 1
                     },
        'sf': {'name': 'SF',
               'color': 'green',
               'linestyle': '-',
               'zorder': 2
               },
        }


for env_domain in env_domain_d.keys():

    nrows = len(N_EPISODES_l)

    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(4, 7))

    for alg in algs:
        for i, n_episodes in enumerate(N_EPISODES_l):
            base_dir = os.path.join(experiment_dir_base,
                                    f'env_domain_{env_domain}',
                                    f'mc_grad_estimator_{alg}',
                                    f'n_episodes_learn_{n_episodes}'
                                    )
            exps_mc = {}
            for (dirpath, dirnames, filenames) in os.walk(base_dir):
                if filenames:
                    exp_name = dirpath
                    params = np.load(os.path.join(exp_name, 'params.npy')).squeeze()
                    true_grads = np.load(os.path.join(exp_name, 'true_grads.npy')).squeeze()
                    grads = np.load(os.path.join(exp_name, 'grads.npy')).squeeze()
                    try:
                        args = json.load(open(os.path.join(exp_name, 'args.json')))
                    except FileNotFoundError:
                        args = {}
                        args['n_actions_per_state'] = exp_name.split('n_actions_per_state')[-1].split('_')[-1].split('/')[0]
                        args['seed'] = exp_name.split('/')[-1]

                    n_actions_per_state = int(args['n_actions_per_state'])
                    if n_actions_per_state not in exps_mc:
                        exps_mc[n_actions_per_state] = {}
                    if args['seed'] not in exps_mc[n_actions_per_state]:
                        exps_mc[n_actions_per_state][args['seed']] = []

                    exps_mc[n_actions_per_state][args['seed']].append(params)
                    exps_mc[n_actions_per_state][args['seed']].append(true_grads)
                    exps_mc[n_actions_per_state][args['seed']].append(grads)

            n_actions_per_state = []
            true_grads = []
            grads = []

            for mc in sorted(exps_mc):
                true_grads_ = []
                grads_ = []
                for seed in exps_mc[mc]:
                    true_grads_.append((exps_mc[mc][seed][1]))
                    grads_.append((exps_mc[mc][seed][2]))

                n_actions_per_state.append(int(mc))
                true_grads.append(np.asarray(true_grads_))
                grads.append(np.asarray(grads_))

            ratio_norms = []
            cos_dists = []
            for true_grad, grad in zip(true_grads, grads):
                ratio_norm = np.abs(np.linalg.norm(grad, axis=-1) - np.linalg.norm(true_grad, axis=-1)) / \
                             np.linalg.norm(true_grad, axis=-1)
                ratio_norms.append(ratio_norm)

                cos_dist_l = []
                for tg, g in zip(true_grad, grad):
                    cos_dist_l.append(cosine(tg, g))
                cos_dists.append(np.asarray(cos_dist_l))

            ratio_norms = np.asarray(ratio_norms).T
            cos_dists = np.asarray(cos_dists).T

            y_mean, y_inf, y_sup = mean_confidence_interval(ratio_norms, confidence=0.95, axis=0)
            axs[i][0].plot(n_actions_per_state, y_mean, linestyle=algs[alg]['linestyle'],
                           color=algs[alg]['color'],
                           zorder=algs[alg]['zorder']
                           )
            axs[i][0].fill_between(n_actions_per_state, y_inf, y_sup, alpha=ALPHA,
                                color=algs[alg]['color'],
                                zorder=algs[alg]['zorder'])

            y_mean, y_inf, y_sup = mean_confidence_interval(cos_dists, confidence=0.95, axis=0)
            axs[i][1].plot(n_actions_per_state, y_mean, linestyle=algs[alg]['linestyle'],
                           color=algs[alg]['color'],
                           zorder=algs[alg]['zorder']
                           )
            axs[i][1].fill_between(n_actions_per_state, y_inf, y_sup, alpha=ALPHA,
                                color=algs[alg]['color'],
                                zorder=algs[alg]['zorder'])

            # axs[i][0].set_ylabel(f'{n_episodes}', rotation='horizontal', loc='top', labelpad=-10)
            axs[i][0].text(0.0, 1.02, f'{n_episodes} trajectories', transform=axs[i][0].transAxes, fontsize=7)

    for i in range(len(axs)):
        axs[i][0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axs[i][1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axs[i][0].set_yscale('log')
        axs[i][1].set_yscale('log')
        axs[i][0].set_xlim(n_actions_per_state[0] - 1, n_actions_per_state[-1] + 1)
        axs[i][1].set_xlim(n_actions_per_state[0] - 1, n_actions_per_state[-1] + 1)
        axs[i][0].set_xticks(np.linspace(n_actions_per_state[0], n_actions_per_state[-1], 5, dtype=int))
        axs[i][1].set_xticks(np.linspace(n_actions_per_state[0], n_actions_per_state[-1], 5, dtype=int))
        if i != len(axs) - 1:
            axs[i][0].tick_params(labelbottom=False)
            axs[i][1].tick_params(labelbottom=False)

        for tick in axs[i][0].yaxis.get_major_ticks():
            tick.label.set_fontsize(12)

    axs[-1][0].set_xlabel('Actions per state', labelpad=1.)
    axs[-1][1].set_xlabel('Actions per state', labelpad=1.)
    axs[0][0].set_title('Relative abs. error', pad=10.0)
    axs[0][1].set_title('Cosine distance', pad=10.0)

    # Save
    fig.tight_layout(h_pad=0.2, w_pad=0.0)
    figname = f'lqr_{env_domain}'
    fig.savefig(f'./out/{figname}.pdf', bbox_inches='tight')

# Legend
for alg in algs:
    axs[0][0].plot([], [], c=algs[alg]['color'], label=algs[alg]['name'])
fig.legend(bbox_to_anchor=(0.5, 0.), loc="center", bbox_transform=fig.transFigure, ncol=3)

export_legend(axs[0][0], filename="out/legend_lqr.pdf")
