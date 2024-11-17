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

launcher = Launcher(exp_name='lqr/lqr_pg_error',
                    python_file='exp_oracle_pg_lqr',
                    # project_name='project01263',
                    n_exp=25,
                    n_cores=1,
                    memory=3000,
                    days=0,
                    hours=12,
                    minutes=59,
                    seconds=59,
                    n_jobs=1,
                    conda_env='ps-mvd',
                    gres='gpu:rtx2080:1' if use_cuda else None,
                    use_timestamp=True)

n_epochs = 1
n_episodes_test = 1
n_epochs_policy = 1
lr_actor = 1e-6
coupling = True
use_cuda = False
debug = False
verbose = False
render = False

env_domain_d = {
                'LQR2states1actions': {'n_actions': 1},
                'LQR2states2actions': {'n_actions': 2},
                'LQR4states4actions': {'n_actions': 4},
                'LQR6states6actions': {'n_actions': 6}
                }

horizon_l = [1000]
gamma_l = [0.99]
initial_std_l = [0.1]
n_episodes_learn_l = [10]
mc_samples_gradient_l = [10]
mc_grad_estimator_l = ['mvd', 'sf', 'reptrick']

noise_q_amp_factor_l = [0.0001, 0.001, 0.01, 0.1]
noise_q_freq_l = [0.01, 0.1, 1.0, 10., 100., 1000.]


for horizon, gamma in zip(horizon_l, gamma_l):

    for env_domain, initial_std, mc_grad_estimator, n_episodes_learn, mc_samples_gradient in \
            product(env_domain_d.keys(), initial_std_l, mc_grad_estimator_l, n_episodes_learn_l, mc_samples_gradient_l):

        n_actions_per_state = mc_samples_gradient * 2 * env_domain_d[env_domain]['n_actions']
        if mc_grad_estimator == 'mvd':
            mc_samples = mc_samples_gradient
        elif mc_grad_estimator == 'sf' or mc_grad_estimator == 'reptrick':
            mc_samples = n_actions_per_state
        else:
            raise NotImplementedError

        for noise_q_freq, noise_q_amp_factor in product(noise_q_freq_l, noise_q_amp_factor_l):

            launcher.add_experiment(env_domain=env_domain,
                                    mc_grad_estimator=mc_grad_estimator,
                                    noise_q_amp_factor=noise_q_amp_factor,
                                    noise_q_freq=noise_q_freq
                                    )

            launcher.add_default_params(
                                        horizon=horizon,
                                        gamma=gamma,
                                        initial_std=initial_std,
                                        n_episodes_learn=n_episodes_learn,
                                        mc_samples_gradient=mc_samples,
                                        n_actions_per_state=n_actions_per_state,
                                        n_epochs=n_epochs,
                                        n_episodes_test=n_episodes_test,
                                        n_epochs_policy=n_epochs_policy,
                                        lr_actor=lr_actor,
                                        coupling=coupling,
                                        use_cuda=use_cuda,
                                        debug=debug,
                                        verbose=verbose,
                                        render=render)

    launcher.run(LOCAL, test)
