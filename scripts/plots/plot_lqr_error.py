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
experiment_dir_base = f'../../logs/lqr_pg_error'

env_domain_l = ['LQR2states1actions', 'LQR2states2actions', 'LQR4states4actions', 'LQR6states6actions']

noise_q_amp_d = {
                 0.0001: {'linestyle': 'dashed'},
                 0.001: {'linestyle': 'dotted'},
                 0.01: {'linestyle': 'dashdot'},
                 0.1: {'linestyle': '-'}
                 }

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

for env_domain in env_domain_l:

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(4, 2.5))

    for alg in algs:
        for q_amp in noise_q_amp_d.keys():
            base_dir = os.path.join(experiment_dir_base,
                                    'env_domain_' + env_domain,
                                    'mc_grad_estimator_' + alg,
                                    f'noise_q_amp_factor_{q_amp}')
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
                        args['noise_q_freq'] = exp_name.split('noise_q_freq')[-1].split('_')[-1].split('/')[0]
                        args['seed'] = exp_name.split('/')[-1]

                    noise_q_freq = float(args['noise_q_freq'])
                    if noise_q_freq not in exps_mc:
                        exps_mc[noise_q_freq] = {}
                    if args['seed'] not in exps_mc[noise_q_freq]:
                        exps_mc[noise_q_freq][args['seed']] = []

                    exps_mc[noise_q_freq][args['seed']].append(params)
                    exps_mc[noise_q_freq][args['seed']].append(true_grads)
                    exps_mc[noise_q_freq][args['seed']].append(grads)

            noise_q_freq = []
            true_grads = []
            grads = []

            for freq in sorted(exps_mc):
                true_grads_ = []
                grads_ = []
                for seed in exps_mc[freq]:
                    true_grads_.append((exps_mc[freq][seed][1]))
                    grads_.append((exps_mc[freq][seed][2]))

                noise_q_freq.append(float(freq))
                true_grads.append(np.asarray(true_grads_))
                grads.append(np.asarray(grads_))

            ratio_norms = []
            cos_sims = []
            for true_grad, grad in zip(true_grads, grads):
                ratio_norm = np.abs(np.linalg.norm(grad, axis=-1) - np.linalg.norm(true_grad, axis=-1)) / \
                             np.linalg.norm(true_grad, axis=-1)
                ratio_norms.append(ratio_norm)

                cos_sim_l = []
                for tg, g in zip(true_grad, grad):
                    cos_sim_l.append(cosine(tg, g))
                cos_sims.append(np.asarray(cos_sim_l))

            ratio_norms = np.asarray(ratio_norms).T
            cos_sims = np.asarray(cos_sims).T

            y_mean, y_inf, y_sup = mean_confidence_interval(ratio_norms, confidence=0.95, axis=0)

            axs[0].plot(noise_q_freq, y_mean,
                        color=algs[alg]['color'],
                        zorder=algs[alg]['zorder'],
                        linestyle=noise_q_amp_d[q_amp]['linestyle']
                        )

            y_mean, y_inf, y_sup = mean_confidence_interval(cos_sims, confidence=0.95, axis=0)
            axs[1].plot(noise_q_freq, y_mean,
                        color=algs[alg]['color'],
                        zorder=algs[alg]['zorder'],
                        linestyle=noise_q_amp_d[q_amp]['linestyle']
                        )

    for ax in axs:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xticks(noise_q_freq)
        set_small_ticks(ax, fontsize=9)
        ax.set_xlabel('Error frequency', labelpad=1.0)

    axs[0].set_title('Relative abs. error', pad=3.0)
    axs[1].set_title('Cosine distance', pad=3.0)

    # Save
    fig.tight_layout(h_pad=0.2, w_pad=0.0)
    figname = f'lqr_error_{env_domain}'
    fig.savefig(f'./out/{figname}.pdf', bbox_inches='tight')

# Legend
for alg in algs:
    axs[0].plot([], [], c=algs[alg]['color'], label=algs[alg]['name'])
for amp in noise_q_amp_d.keys():
    axs[0].plot([], [], c='black', label=f'$\\alpha = {amp}$', linestyle=noise_q_amp_d[amp]['linestyle'])
fig.legend(bbox_to_anchor=(0.5, 0.), loc="center", bbox_transform=fig.transFigure, ncol=3)
export_legend(axs[0], filename="out/legend_lqr_error.pdf")
