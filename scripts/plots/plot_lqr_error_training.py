import json
import os

from basic_settings import *

mpl.rcParams['lines.linewidth'] = 1.
mpl.rcParams['text.usetex'] = True
plt.rcParams["axes.grid.axis"] = "both"
plt.rcParams["axes.grid"] = True
ALPHA = 0.2

########################################################################################################################
experiment_dir_base = f'../../logs/lqr_pg_error_training'

env_domain_d = {'LQR2states1actions': {'ylims': [-4900, -850]},
                'LQR2states2actions': {'ylims': [-11000, -850]},
                'LQR4states4actions': {'ylims': [-400000, -1300]},
                'LQR6states6actions': {'ylims': [-55000, -1300]}
                }

noise_freq_d = {
                1.0: {'linestyle': '-'},
                10.0: {'linestyle': 'dotted'},
                100.0: {'linestyle': 'dashed'}
                }

noise_q_amp_factor_l = [0.0, 0.001, 0.01]

algs = {
        'mvd': {'name': 'MVD',
                'color': 'blue',
                'linestyle': '-',
                'zorder': 3
                },
        'sf': {'name': 'SF',
               'color': 'green',
               'linestyle': '-',
               'zorder': 2
               },
        'reptrick': {'name': 'RepTrick',
                     'color': 'orange',
                     'linestyle': '-',
                     'zorder': 1
                     },
        }

for env_domain in env_domain_d.keys():

    for i, alg in enumerate(algs):

        fig, axs = plt.subplots(nrows=len(noise_q_amp_factor_l),
                                ncols=1,
                                figsize=(4.6, 4.5 if len(noise_q_amp_factor_l) >= 3 else 3.),
                                dpi=300)

        for j, q_amp_factor in enumerate(noise_q_amp_factor_l):
            exps = {}
            for noise_q_freq in noise_freq_d.keys():
                base_dir = os.path.join(experiment_dir_base,
                                        f'env_domain_{env_domain}',
                                        f'mc_grad_estimator_{alg}',
                                        f'noise_q_amp_factor_{q_amp_factor}',
                                        f'noise_q_freq_{noise_q_freq}')
                for (dirpath, dirnames, filenames) in os.walk(base_dir):
                    if filenames:
                        exp_name = dirpath
                        J = np.load(os.path.join(exp_name, 'Jgamma.npy')).squeeze()
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
                        if noise_q_freq not in exps:
                            exps[noise_q_freq] = {}
                        if args['seed'] not in exps[noise_q_freq]:
                            exps[noise_q_freq][args['seed']] = []

                        exps[noise_q_freq][args['seed']].append(params)
                        exps[noise_q_freq][args['seed']].append(true_grads)
                        exps[noise_q_freq][args['seed']].append(grads)
                        exps[noise_q_freq][args['seed']].append(J)

            for k, freq in enumerate(sorted(exps.keys())):
                Js = []
                for seed in exps[freq]:
                    Js.append((exps[freq][seed][3]))

                n_samples = np.asarray(Js)[0, :, 0]
                Js = np.asarray(Js)[:, :, 1]

                y_mean, y_inf, y_sup = mean_confidence_interval(Js, confidence=0.95, axis=0)
                axs[j].plot(n_samples, y_mean,
                            color=algs[alg]['color'],
                            zorder=algs[alg]['zorder'],
                            linestyle=noise_freq_d[freq]['linestyle']
                            )
                axs[j].fill_between(n_samples, y_inf, y_sup, alpha=ALPHA,
                                    color=algs[alg]['color'],
                                    zorder=algs[alg]['zorder']
                                    )

                axs[j].text(0.0, 1.02, f'${{\\alpha = {q_amp_factor}}}$',
                            transform=axs[j].transAxes, fontsize=7)

        ylims = env_domain_d[env_domain]['ylims']
        for j, ax in enumerate(axs):
            ax.set_ylim(ylims[0], ylims[1])
            ax.set_yscale('symlog')
            ax.set_xlabel('')
            if j != len(axs) - 1:
                ax.set_xticklabels([])
        axs[-1].set_xlabel('Steps')
        axs[-1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        # Save
        fig.tight_layout(h_pad=0.0, w_pad=0.4)
        figname = f'lqr_error_training_{env_domain}_{alg}'
        fig.savefig(f'./out/{figname}.pdf', bbox_inches='tight')

# Legend
for i, alg in enumerate(algs):
    axs[0].plot([], [], c=algs[alg]['color'], label=algs[alg]['name'])

for freq in sorted(noise_freq_d.keys()):
    axs[0].plot([], [], c='black', label=f'$f = {freq}$', linestyle=noise_freq_d[freq]['linestyle'])

fig.legend(bbox_to_anchor=(0.5, 0.), loc="center", bbox_transform=fig.transFigure, ncol=3)
export_legend(axs[0], filename="out/legend_lqr_error_training.pdf")
