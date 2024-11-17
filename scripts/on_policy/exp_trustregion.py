import argparse
import json
import os

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from mushroom_rl.utils.callbacks import PlotDataset, CollectDataset

from mushroom_rl.core import Core
from mushroom_rl.environments import Gym, DMControl
from mushroom_rl.algorithms.actor_critic import TRPO, PPO

from mushroom_rl.policy import GaussianTorchPolicy
from mushroom_rl.utils.dataset import compute_J

from mushroom_rl.utils.preprocessors import StandardizationPreprocessor, MinMaxPreprocessor

from scripts.on_policy.display import visualization
from scripts.on_policy.exp_tree_pg import ActorNetwork1
from src.envs.corridor import Corridor
from src.envs.room import Room
from src.utils.seeds import fix_random_seed


########################################################################################################################
def experiment(alg, env_domain, env_task, horizon, gamma,
               n_epochs, n_steps,
               n_steps_per_fit,
               n_episodes_test,
               n_features_critic,
               lr_critic,
               batch_size_critic,
               n_epochs_critic,
               initial_std,
               n_epochs_actor,
               batch_size_actor,
               n_features_actor,
               lr_actor,
               preprocess_states,
               use_cuda,
               debug,
               render,
               verbose,
               callbacks,
               seed, results_dir):

    if use_cuda:
        torch.set_num_threads(1)

    print('Env id: {}, Alg: {}'.format(env_domain, alg))

    # Create results directory
    results_dir = os.path.join(results_dir, str(seed))
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    if env_domain == 'Room':
        mdp = Room(horizon=horizon, gamma=gamma)
    elif env_domain == 'Corridor':
        mdp = Corridor(horizon=horizon, gamma=gamma)
    else:
        if env_task != 'NA':
            mdp = DMControl(env_domain, env_task, horizon, gamma)
        else:
            mdp = Gym(env_domain, horizon, gamma)

    # Fix seed
    fix_random_seed(seed, mdp)

    # Approximators
    policy_params = dict(
        std_0=initial_std,
        n_features=n_features_actor,
        use_cuda=use_cuda
    )
    ppo_params = dict(actor_optimizer={'class': optim.Adam,
                                       'params': {'lr': lr_actor}},
                      n_epochs_policy=n_epochs_actor,
                      batch_size=batch_size_actor,
                      eps_ppo=.2,
                      lam=.95)

    trpo_params = dict(
                       ent_coeff=0.0,
                       max_kl=.01,
                       lam=.95,
                       n_epochs_line_search=10,
                       n_epochs_cg=100,
                       cg_damping=1e-2,
                       cg_residual_tol=1e-10)

    critic_params = dict(network=ActorNetwork1,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic}},
                         loss=F.mse_loss,
                         n_features=n_features_critic,
                         batch_size=batch_size_critic,
                         input_shape=mdp.info.observation_space.shape,
                         output_shape=(1,))

    policy = GaussianTorchPolicy(ActorNetwork1,
                                 mdp.info.observation_space.shape,
                                 mdp.info.action_space.shape,
                                 **policy_params)

    ppo_params['critic_params'] = critic_params

    trpo_params['critic_params'] = critic_params

    # Agent
    if alg == 'ppo':
        agent = PPO(mdp.info, policy,
                    **ppo_params,
                    critic_fit_params=dict(n_epochs=n_epochs_critic)
                    )
    elif alg == 'trpo':
        agent = TRPO(mdp.info, policy,
                     **trpo_params,
                     critic_fit_params=dict(n_epochs=n_epochs_critic)
                     )
    else:
        raise NotImplementedError

    # Algorithm
    prepro = None
    if preprocess_states == 'minmax':
        prepro = MinMaxPreprocessor(mdp_info=mdp.info)
    elif preprocess_states == 'standard':
        prepro = StandardizationPreprocessor(mdp_info=mdp.info)

    plotter = None
    dataset_callback = None
    if debug:
        plotter = PlotDataset(mdp.info, obs_normalized=True if preprocess_states != None else False)

    visualization_callback = None
    if callbacks:
        dataset_callback = CollectDataset()
        visualization_callback = visualization(env_domain, agent._V, results_dir, n_episodes_test, debug)

    core = Core(agent, mdp,
                callback_step=plotter,
                preprocessors=[prepro] if prepro is not None else None,
                callbacks_fit=[dataset_callback] if dataset_callback else None)

    # TRAIN
    Jgamma_l = []
    R_l = []
    entropy_l = []

    # First evaluation
    dataset = core.evaluate(n_episodes=n_episodes_test, render=render)
    Jgamma = compute_J(dataset, gamma)
    R = compute_J(dataset)
    entropy = agent.policy.entropy()
    print('J: {:.4f} - R: {:.4f} - H(pi): {:.6f}'.format(np.mean(Jgamma), np.mean(R), entropy))
    Jgamma_l.append((0, np.mean(Jgamma)))
    R_l.append((0, np.mean(R)))
    entropy_l.append((0, entropy))

    for n in range(n_epochs):
        print(f'-> Epoch: {n}/{n_epochs-1}, #samples: {(n+1)*n_steps}')

        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=not verbose)

        if dataset_callback:
            dataset_callback.clean()

        dataset = core.evaluate(n_episodes=n_episodes_test, render=True if n % 10 == 0 and render else False,
                                quiet=not verbose)

        if visualization_callback:
            dataset_ = CollectDataset()
            dataset_(dataset)
            visualization_callback(dataset_, n)

        Jgamma = compute_J(dataset, gamma)
        R = compute_J(dataset)
        entropy = agent.policy.entropy()
        print('J: {:.4f} - R: {:.4f} - H(pi): {:.6f}'.format(np.mean(Jgamma), np.mean(R), entropy))
        Jgamma_l.append((n_steps*(n+1), np.mean(Jgamma)))
        R_l.append((n_steps*(n+1), np.mean(R)))
        entropy_l.append((n_steps*(n+1), entropy))

        # Save agent and results
        if n % max(1, int(n_epochs * 0.05)) == 0 or n == n_epochs - 1:
            np.save(os.path.join(results_dir, 'Jgamma.npy'), np.array(Jgamma_l))
            np.save(os.path.join(results_dir, 'R.npy'), np.array(R_l))
            np.save(os.path.join(results_dir, 'entropy.npy'), np.array(entropy_l))

    agent.save(os.path.join(results_dir, f'agent_{n_epochs}.msh'))


def default_params():
    defaults = dict(
        alg='ppo',
        env_domain='Pendulum-v0',
        env_task='NA',
        horizon=200,
        gamma=0.99,
        n_epochs=200,
        n_steps=3000,
        n_steps_per_fit=3000,
        n_episodes_test=5,
        n_features_critic=32,
        lr_critic=3e-4,
        batch_size_critic=64,
        n_epochs_critic=10,
        initial_std=1.0,
        n_epochs_actor=4,
        batch_size_actor=64,
        n_features_actor=32,
        lr_actor=1e-4,
        preprocess_states='None',
        use_cuda=False,
        debug=False,
        verbose=False,
        render=False,
        callbacks=False,
        seed=1,
        results_dir='/tmp/results'
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--alg', type=str)
    parser.add_argument('--env-domain', type=str)
    parser.add_argument('--env-task', type=str)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--n-steps', type=int)
    parser.add_argument('--n-steps-per-fit', type=int)
    parser.add_argument('--n-episodes-test', type=int)
    parser.add_argument('--n-features-critic', type=int)
    parser.add_argument('--lr-critic', type=float)
    parser.add_argument('--batch-size-critic', type=int)
    parser.add_argument('--n-epochs-critic', type=int)
    parser.add_argument('--initial-std', type=float)
    parser.add_argument('--n-epochs-actor', type=int)
    parser.add_argument('--batch-size-actor', type=int)
    parser.add_argument('--n-features-actor', type=int)
    parser.add_argument('--lr-actor', type=float)
    parser.add_argument('--preprocess-states', type=str)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--callbacks', action='store_true')
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
