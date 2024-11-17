import socket
from experiment_launcher import Launcher
import torch
import numpy as np
from itertools import product


hostname = socket.gethostname()
LOCAL = False if hostname == 'mn01' or 'hla' in hostname else True

# Fix number of torch threads
if LOCAL:
    torch.set_num_threads(1)


test = False
use_cuda = False

launcher = Launcher(exp_name='lqr_pg_error_training',
                    python_file='exp_oracle_pg_lqr',
                    # project_name='project01263',
                    n_exp=25,
                    n_cores=1,
                    memory=5000,
                    days=0,
                    hours=12,
                    minutes=59,
                    seconds=59,
                    n_jobs=3,
                    conda_env='ps-mvd',
                    gres='gpu:rtx2080:1' if use_cuda else None,
                    use_timestamp=True)

n_episodes_test = 1
n_epochs_policy = 1
coupling = True
use_cuda = False
debug = False
verbose = False
render = False

env_domain_d = {
                'LQR2states1actions': {'n_actions': 1, 'lr_actor': 5e-2, 'n_epochs': 50},
                'LQR2states2actions': {'n_actions': 2, 'lr_actor': 1e-2, 'n_epochs': 200},
                'LQR4states4actions': {'n_actions': 4, 'lr_actor': 3e-3, 'n_epochs': 1500},
                'LQR6states6actions': {'n_actions': 6, 'lr_actor': 5e-3, 'n_epochs': 1000},
                }

horizon = 1000
gamma = 0.99
initial_std = 0.1
n_episodes_learn = 1
mc_samples_gradient = 1
mc_grad_estimator_l = ['mvd', 'sf', 'reptrick']


noise_q_amp_factor_l = [0.0, 0.001, 0.01]
noise_q_freq_l = [1.0, 10., 100.]


for env_domain, mc_grad_estimator, noise_q_amp_factor, noise_q_freq in \
    product(env_domain_d.keys(), mc_grad_estimator_l, noise_q_amp_factor_l, noise_q_freq_l):

    n_actions_per_state = mc_samples_gradient * 2 * env_domain_d[env_domain]['n_actions']
    if mc_grad_estimator == 'mvd':
        mc_samples = mc_samples_gradient
    elif mc_grad_estimator == 'sf' or mc_grad_estimator == 'reptrick':
        mc_samples = n_actions_per_state
    else:
        raise NotImplementedError

    launcher.add_experiment(env_domain=env_domain,
                            mc_grad_estimator=mc_grad_estimator,
                            noise_q_amp_factor=noise_q_amp_factor,
                            noise_q_freq=noise_q_freq
                            )

    launcher.add_default_params(noise_a_type='random_p',
                                horizon=horizon,
                                gamma=gamma,
                                initial_std=initial_std,
                                mc_grad_estimator=mc_grad_estimator,
                                n_episodes_learn=n_episodes_learn,
                                mc_samples_gradient=mc_samples,
                                n_actions_per_state=n_actions_per_state,
                                n_epochs=env_domain_d[env_domain]['n_epochs'],
                                n_episodes_test=n_episodes_test,
                                n_epochs_policy=n_epochs_policy,
                                lr_actor=env_domain_d[env_domain]['lr_actor'],
                                coupling=coupling,
                                use_cuda=use_cuda,
                                debug=debug,
                                verbose=verbose,
                                render=render)

    launcher.run(LOCAL, test)
