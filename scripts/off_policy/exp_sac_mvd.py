import os
import argparse
import json

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.core import Core
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.callbacks import PlotDataset
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor

from src.algs.step_based.sac_pg import SAC_PolicyGradient
from src.utils.seeds import fix_random_seed

from exp_sac import CriticNetwork, ActorNetwork

########################################################################################################################
def experiment(env_id, horizon, gamma,
               n_epochs, n_steps, n_episodes_test,
               initial_replay_size,
               max_replay_size,
               batch_size,
               warmup_transitions,
               tau,
               lr_alpha,
               n_features_actor,
               n_features_critic,
               lr_actor,
               lr_critic,
               mc_samples_gradient,
               coupling,
               preprocess_states,
               use_cuda,
               debug,
               verbose,
               seed, results_dir):

    if use_cuda:
        torch.set_num_threads(1)

    print('Env id: {}, Alg: SAC-MVD'.format(env_id))

    # Create results directory
    results_dir = os.path.join(results_dir, str(seed))
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    mdp = Gym(env_id, horizon, gamma)

    # Fix seed
    fix_random_seed(seed, mdp)

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_mu_params = dict(network=ActorNetwork,
                           n_features=n_features_actor,
                           input_shape=actor_input_shape,
                           output_shape=mdp.info.action_space.shape,
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=ActorNetwork,
                              n_features=n_features_actor,
                              input_shape=actor_input_shape,
                              output_shape=mdp.info.action_space.shape,
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': lr_actor}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic}},
                         loss=F.mse_loss,
                         n_features=n_features_critic,
                         input_shape=critic_input_shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # Agent
    mc_gradient_estimator = {'estimator': 'mvd',
                             'n_samples': mc_samples_gradient,
                             'coupling': coupling}
    agent = SAC_PolicyGradient(mdp.info, actor_mu_params, actor_sigma_params,
                               actor_optimizer, critic_params,
                               batch_size, initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha,
                               critic_fit_params=None,
                               mc_gradient_estimator=mc_gradient_estimator
                               )

    # Algorithm
    prepro = None
    if preprocess_states:
        # prepro = MinMaxPreprocessor(mdp_info=mdp.info)
        prepro = StandardizationPreprocessor(mdp_info=mdp.info)

    plotter = None
    if debug:
        plotter = PlotDataset(mdp.info, obs_normalized=True)

    core = Core(agent, mdp,
                callback_step=plotter,
                preprocessors=[prepro] if prepro is not None else None)

    # TRAIN
    Jgamma_l = []
    Jenv_l = []

    # Fill up the replay memory
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size)

    # First evaluation
    dataset = core.evaluate(n_episodes=n_episodes_test, render=False)
    Jgamma = compute_J(dataset, gamma)
    Jenv = compute_J(dataset)
    print('J: {:.4f} - Jenv: {:.4f}'.format(np.mean(Jgamma), np.mean(Jenv)))
    Jgamma_l.append((0, np.mean(Jgamma)))
    Jenv_l.append((0, np.mean(Jenv)))

    for n in range(n_epochs):
        print('Epoch: {}/{}'.format(n, n_epochs - 1))

        core.learn(n_steps=n_steps, n_steps_per_fit=1, quiet=not verbose)

        dataset = core.evaluate(n_episodes=n_episodes_test, render=True if n % 10 == 0 and debug else False,
                                quiet=not verbose)
        Jgamma = compute_J(dataset, gamma)
        Jenv = compute_J(dataset)
        print('J: {:.4f} - Jenv: {:.4f}'.format(np.mean(Jgamma), np.mean(Jenv)))
        Jgamma_l.append((n_steps*(n+1), np.mean(Jgamma)))
        Jenv_l.append((n_steps*(n+1), np.mean(Jenv)))

        if n % max(1, int(n_epochs * 0.05)) == 0 or n == n_epochs - 1:
            np.save(os.path.join(results_dir, 'J.npy'), np.array(Jgamma_l))
            np.save(os.path.join(results_dir, 'R.npy'), np.array(Jenv_l))


def default_params():
    defaults = dict(
        env_id='Hopper-v2',
        horizon=1000, gamma=0.99,
        n_epochs=200, n_steps=5000, n_episodes_test=10,
        initial_replay_size=5000,
        max_replay_size=500000,
        batch_size=256,
        warmup_transitions=5000,
        tau=0.005,
        lr_alpha=3e-4,
        n_features_actor=256,
        n_features_critic=256,
        lr_actor=5e-5,
        lr_critic=3e-4,
        mc_samples_gradient=1,
        coupling=False,
        preprocess_states=False,
        use_cuda=False,
        debug=False,
        verbose=False,
        seed=1, results_dir='/tmp/results'
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-id', type=str)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--n-steps', type=int)
    parser.add_argument('--n-episodes-test', type=int)
    parser.add_argument('--initial-replay-size', type=int)
    parser.add_argument('--max-replay-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--warmup-transitions', type=int)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--lr-alpha', type=float)
    parser.add_argument('--n-features-actor', type=int)
    parser.add_argument('--n-features-critic', type=int)
    parser.add_argument('--lr-actor', type=float)
    parser.add_argument('--lr-critic', type=float)
    parser.add_argument('--mc-samples-gradient', type=int)
    parser.add_argument('--coupling', action='store_true')
    parser.add_argument('--preprocess-states', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--results-dir', type=str)

    parser.set_defaults(**default_params())
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    args = parse_args()

    # Save args
    print(args)
    os.makedirs(os.path.join(args['results_dir'], str(args['seed'])), exist_ok=True)
    with open(os.path.join(args['results_dir'], str(args['seed']), 'args.json'), 'w') as f:
        json.dump(args, f, indent=2)

    experiment(**args)

