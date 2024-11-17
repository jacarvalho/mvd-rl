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


launcher = Launcher(exp_name='on_policy_tree_mvd',
                    python_file='exp_tree_pg',
                    # project_name='project01263',
                    n_exp=25,
                    n_cores=1,
                    memory=2000,
                    days=0,
                    hours=23,
                    minutes=59,
                    seconds=59,
                    n_jobs=7,
                    conda_env='ps-mvd',
                    gres='gpu:rtx2080:1' if use_cuda else None,
                    use_timestamp=True)


env_domain_l = ['Pendulum-v0']
horizon_l = [200]
gamma_l = [0.99]
n_epochs_l = [100]
n_steps_l = [3000]
n_steps_per_fit_l = n_steps_l
n_episodes_test = 10
n_iters_critic_l = [100]
mc_samples_action_next_l = [1]
n_estimators_tree_fit_l = [100]
min_samples_split_tree_fit_l = [2]
min_samples_leaf_tree_fit_l = [1]
n_jobs_tree_fit = 1
use_replay_memory = True
max_replay_memory_size = 500000
replay_memory_samples_l = [25000]
n_epochs_policy_l = [4]
mc_grad_estimator = 'mvd'
batch_size_policy_l = [256]
n_features_actor_l = [32]
lr_actor_l = [3e-4]
mc_samples_gradient_l = [1]
coupling = True
squashed_policy_l = [True]
initial_std = 1.0

callbacks = True
verbose = False
preprocess_states_l = [False]

for env_domain, horizon, gamma, n_epochs, n_steps_per_fit, \
    mc_samples_action_next, \
    min_samples_split_tree_fit, min_samples_leaf_tree_fit,\
    squashed_policy, preprocess_states, n_steps, n_estimators_tree_fit, replay_memory_samples,\
    batch_size_policy, n_features_actor, lr_actor, mc_samples_gradient, n_iters_critic, n_epochs_policy in \
        zip(env_domain_l, horizon_l, gamma_l, n_epochs_l, n_steps_per_fit_l,
            mc_samples_action_next_l,
            min_samples_split_tree_fit_l, min_samples_leaf_tree_fit_l,
            squashed_policy_l, preprocess_states_l,
            n_steps_l,
            n_estimators_tree_fit_l,
            replay_memory_samples_l,
            batch_size_policy_l,
            n_features_actor_l, lr_actor_l, mc_samples_gradient_l, n_iters_critic_l,
            n_epochs_policy_l):

    launcher.add_default_params(horizon=horizon,
                                gamma=gamma,
                                n_epochs=n_epochs,
                                n_steps_per_fit=n_steps_per_fit,
                                n_episodes_test=n_episodes_test,
                                n_jobs_tree_fit=n_jobs_tree_fit,
                                mc_grad_estimator=mc_grad_estimator,
                                coupling=coupling,
                                initial_std=initial_std,
                                callbacks=callbacks,
                                verbose=verbose,
                                use_replay_memory=use_replay_memory,
                                max_replay_memory_size=max_replay_memory_size,
                                mc_samples_action_next=mc_samples_action_next,
                                min_samples_split_tree_fit=min_samples_split_tree_fit,
                                min_samples_leaf_tree_fit=min_samples_leaf_tree_fit,
                                squashed_policy=squashed_policy,
                                preprocess_states=preprocess_states,
                                n_steps=n_steps,
                                n_iters_critic=n_iters_critic,
                                n_estimators_tree_fit=n_estimators_tree_fit,
                                replay_memory_samples=replay_memory_samples,
                                batch_size_policy=batch_size_policy,
                                n_features_actor=n_features_actor,
                                lr_actor=lr_actor,
                                n_epochs_policy=n_epochs_policy,
                                mc_samples_gradient=mc_samples_gradient
                                )

    launcher.add_experiment(env_domain=env_domain)

    launcher.run(LOCAL, test)

