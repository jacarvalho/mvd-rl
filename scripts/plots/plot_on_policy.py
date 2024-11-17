import os

from basic_settings import *

mpl.rcParams['lines.linewidth'] = 1.
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.size'] = 15

########################################################################################################################
experiment_dir_base = '../../logs/'

envs = {
        'env_domain_Pendulum-v0': {'name': 'Pendulum',
                                   'lims_x': [None, None], 'lims_y': [None, None]
                                   },
        'env_domain_LunarLanderContinuous-v2': {'name': 'LunarLander',
                                                'lims_x': [None, None], 'lims_y': [None, None]
                                                },
        'env_domain_Corridor': {'name': 'Corridor',
                                'lims_x': [None, None], 'lims_y': [None, None]
                                },
        'env_domain_Room': {'name': 'Room',
                            'lims_x': [None, None], 'lims_y': [None, None]
                            },
        }

algs = {
        'tree_mvd': {'name': 'Tree-MVD',
                     'color': 'blue',
                     'linestyle': '-',
                     'zorder': 2
                     },
        'ppo': {'name': 'PPO',
                'color': 'red',
                'linestyle': '-',
                'zorder': 1
                },
        'trpo': {'name': 'TRPO',
                 'color': 'orange',
                 'linestyle': '-',
                 'zorder': 0
                 },
        }

FNAME = 'R'
if FNAME == 'R':
    y_label = 'Average Reward'
else:
    y_label = 'Return'

for i, env in enumerate(envs):
    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=(3, 3))

    for alg in algs:
        base_dir = os.path.join(experiment_dir_base, f'on_policy_{alg}', env)
        exp_files_dir = {}
        for (dirpath, dirnames, filenames) in os.walk(base_dir):
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                exp_name = os.path.join(*(file_path.split('/')[:-2]))
                if FNAME in file:
                    np_file = np.load(file_path)
                    if exp_name in exp_files_dir:
                        pass
                    else:
                        exp_files_dir[exp_name] = list()
                    exp_files_dir[exp_name].append(np_file)

        for exp in exp_files_dir:
            n_evals_f_evals = np.asarray([j for j in exp_files_dir[exp]])
            min_steps = min(np.array([i.shape for i in n_evals_f_evals])[:, 0])
            n_evals_f_evals = np.asarray([j[:min_steps] for j in exp_files_dir[exp]])

            n_steps = n_evals_f_evals[0, :, 0]
            ys = n_evals_f_evals[:, :, 1]

            y_mean, y_inf, y_sup = mean_confidence_interval(ys, confidence=0.95, axis=0)

            line, = axs.plot(
                                n_steps, y_mean, linestyle=algs[alg]['linestyle'],
                                color=algs[alg]['color'],
                                zorder=algs[alg]['zorder']
                                )
            axs.fill_between(
                                n_steps, y_inf, y_sup, alpha=0.35,
                                zorder=algs[alg]['zorder'],
                                color=algs[alg]['color'])

    axs.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    axs.set_xlim(envs[env]['lims_x'])
    axs.set_ylim(envs[env]['lims_y'])

    axs.set_xlabel('Steps', fontsize=12)
    axs.set_ylabel(y_label, labelpad=0., fontsize=12)

    # Save
    fig.tight_layout(h_pad=0.0, w_pad=0.9)
    fig.savefig(f'./out/on_policy_{env}.pdf', bbox_inches='tight')

# Legend
for alg in algs:
    axs.plot([], [], c=algs[alg]['color'], label=algs[alg]['name'], linestyle=algs[alg]['linestyle'])
fig.legend(bbox_to_anchor=(0.5, 0.), loc="center", bbox_transform=fig.transFigure, ncol=len(algs), frameon=False)
export_legend(axs, filename="out/legend_on_policy.pdf")

