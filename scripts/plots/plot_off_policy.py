import os

from basic_settings import *

mpl.rcParams['lines.linewidth'] = 1.
mpl.rcParams['text.usetex'] = True

########################################################################################################################
experiment_dir_base = '../../logs/'

algs = {
        'sac_mvd': {'name': 'SAC-MVD',
                    'color': 'blue',
                    'linestyle': '-',
                    'zorder': 7,
                    },
        'sac': {'name': 'SAC',
                'color': 'orange',
                'linestyle': '-',
                'zorder': 6
                },
        'sac_extra_samples': {'name': 'SAC-extra-samples',
                              'color': 'darkorange',
                              'linestyle': 'dashed',
                              'zorder': 5
                              },
        'sac_sf': {'name': 'SAC-SF',
                   'color': 'green',
                   'linestyle': '-',
                   'zorder': 4
                   },
        'sac_sf_extra_samples': {'name': 'SAC-SF-extra-samples',
                                 'color': 'seagreen',
                                 'linestyle': 'dashed',
                                 'zorder': 3
                                 },
        'ddpg': {'name': 'DDPG',
                 'color': 'violet',
                 'linestyle': '-',
                 'zorder': 2
                 },
        'td3': {'name': 'TD3',
                'color': 'salmon',
                'linestyle': '-',
                'zorder': 1
                },
        }


envs = {
        'env_id_AntBulletEnv-v0': {'lims_x': [None, None], 'lims_y': [None, None]},
        'env_id_HalfCheetahBulletEnv-v0': {'lims_x': [None, None], 'lims_y': [None, None]},
        'env_id_HopperBulletEnv-v0': {'lims_x': [None, None], 'lims_y': [None, None]},
        'env_id_Walker2DBulletEnv-v0': {'lims_x': [None, None], 'lims_y': [None, None]},
        'env_id_InvertedPendulumBulletEnv-v0': {'lims_x': [None, None], 'lims_y': [None, None]},
        'env_id_InvertedPendulumSwingupBulletEnv-v0': {'lims_x': [None, None], 'lims_y': [None, None]},
        'env_id_Pendulum-v0': {'lims_x': [None, None], 'lims_y': [None, None]},
        'env_id_ReacherBulletEnv-v0': {'lims_x': [None, None], 'lims_y': [None, None]},
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
        seeds_dir = os.path.join(experiment_dir_base, 'off_policy_' + alg, env)
        n_evals_f_evals = []
        for (dirpath, dirnames, filenames) in os.walk(seeds_dir):
            for file in filenames:
                if FNAME in file:
                    file_path = os.path.join(dirpath, file)
                    np_file = np.load(file_path)
                    n_evals_f_evals.append(np_file)

        min_steps = min(np.array([i.shape for i in n_evals_f_evals])[:, 0])
        n_evals_f_evals = np.asarray([j[:min_steps] for j in n_evals_f_evals])
        n_evals_f_evals = np.asarray(n_evals_f_evals)
        n_steps = n_evals_f_evals[0, :, 0]
        Js = n_evals_f_evals[:, :, 1]

        J_mean, J_inf, J_sup = mean_confidence_interval(Js, confidence=0.95, axis=0)

        line, = axs.plot(n_steps, J_mean,
                         color=algs[alg]['color'],
                         linestyle=algs[alg]['linestyle'],
                         zorder=algs[alg]['zorder']
                         )
        axs.fill_between(n_steps, J_inf, J_sup, color=algs[alg]['color'], alpha=0.1,
                                         zorder=algs[alg]['zorder'])

    env_name = env.split('env_id_')[-1].split('BulletEnv')[0]
    axs.set_title(env_name, y=1.1, pad=0.)

    axs.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
    axs.set_xlim(envs[env]['lims_x'])
    axs.set_ylim(envs[env]['lims_y'])

    axs.set_xlabel('Steps')
    axs.set_ylabel(y_label)

    # Save
    fig.tight_layout(h_pad=0.1, w_pad=0.)
    fig.savefig(f'./out/off_policy_{env_name}.pdf', bbox_inches='tight')

# Legend
for alg in algs:
    axs.plot([], [], c=algs[alg]['color'], label=algs[alg]['name'], linestyle=algs[alg]['linestyle'])
fig.legend(bbox_to_anchor=(0.5, 0.), loc="center", bbox_transform=fig.transFigure, ncol=len(algs), frameon=False)

export_legend(axs, filename="out/legend_off_policy.pdf")
