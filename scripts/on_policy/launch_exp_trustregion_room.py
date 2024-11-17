import socket
from itertools import product

from experiment_launcher import Launcher
import torch

hostname = socket.gethostname()
LOCAL = False if hostname == 'mn01' or 'hla' in hostname else True

# Fix number of torch threads
if LOCAL:
    torch.set_num_threads(1)

test = False
use_cuda = True if hostname in ['johnson'] else False

env_domain_l = ['Corridor', 'Room']
horizon_l = [300]
gamma_l = [0.99]
n_epochs_l = [60]
n_steps_l = [2000]
n_steps_per_fit_l = n_steps_l
n_episodes_test = 10
n_features_critic = 128
lr_critic = 3e-4
batch_size_critic = 64
n_epochs_actor_l = [4]
batch_size_actor_l = [128]
n_features_actor_l = [32]
lr_actor_l = [3e-4]
initial_std = 1.0

callbacks = True
verbose = False
preprocess_states = 'minmax'

alg_l = ['ppo', 'trpo']

for alg in alg_l:

    launcher = Launcher(exp_name=f'on_policy_{alg}',
                        python_file='exp_trustregion',
                        # project_name='project01263',
                        n_exp=25,
                        n_cores=1,
                        memory=2000,
                        days=0,
                        hours=23,
                        minutes=59,
                        seconds=59,
                        n_jobs=1,
                        conda_env='ps-mvd',
                        gres='gpu:rtx2080:1' if use_cuda else None,
                        use_timestamp=True)

    for env_domain in env_domain_l:

        for horizon, gamma, n_epochs, n_steps_per_fit, n_steps, n_features_actor, \
            batch_size_actor, lr_actor, n_epochs_actor in \
                zip(horizon_l, gamma_l, n_epochs_l, n_steps_per_fit_l,
                    n_steps_l, n_features_actor_l, batch_size_actor_l, lr_actor_l, n_epochs_actor_l):

            launcher.add_default_params(alg=alg,
                                        horizon=horizon,
                                        gamma=gamma,
                                        n_epochs=n_epochs,
                                        n_steps_per_fit=n_steps_per_fit,
                                        n_episodes_test=n_episodes_test,
                                        n_features_critic=n_features_critic,
                                        lr_critic=lr_critic,
                                        batch_size_critic=batch_size_critic,
                                        initial_std=initial_std,
                                        callbacks=callbacks,
                                        verbose=verbose,
                                        preprocess_states=preprocess_states,
                                        use_cuda=use_cuda,
                                        n_steps=n_steps,
                                        n_features_actor=n_features_actor,
                                        batch_size_actor=batch_size_actor,
                                        lr_actor=lr_actor,
                                        n_epochs_actor=n_epochs_actor,
                                        )

            launcher.add_experiment(env_domain=env_domain,
                                    )

            launcher.run(LOCAL, test)
