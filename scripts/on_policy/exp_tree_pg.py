import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mushroom_rl.approximators import Regressor
from mushroom_rl.core import Core
from mushroom_rl.environments import DMControl
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.utils.callbacks import CollectDataset
from mushroom_rl.utils.callbacks import PlotDataset
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor, MinMaxPreprocessor
from sklearn.ensemble import ExtraTreesRegressor

from scripts.on_policy.display import visualization, ValueFunction
from src.algs.step_based.trees_pg import RegressionTrees_PolicyGradient
from src.envs.corridor import Corridor
from src.envs.room import Room
from src.policies.torch_policies import GaussianTorchPolicyExtended, GaussianTorchPolicyExtendedSquashed
from src.utils.dtype_utils import NP_FLOAT_DTYPE
from src.utils.seeds import fix_random_seed


########################################################################################################################
class ActorNetwork1(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork1, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a


def experiment(env_domain, env_task, horizon, gamma,
               n_epochs, n_steps,
               n_steps_per_fit,
               n_episodes_test,
               n_iters_critic,
               mc_samples_action_next,
               n_estimators_tree_fit,
               min_samples_split_tree_fit,
               min_samples_leaf_tree_fit,
               n_jobs_tree_fit,
               use_replay_memory,
               max_replay_memory_size,
               replay_memory_samples,
               initial_std,
               squashed_policy,
               n_epochs_policy,
               batch_size_policy,
               n_features_actor,
               lr_actor,
               mc_grad_estimator,
               mc_samples_gradient,
               coupling,
               preprocess_states,
               use_cuda,
               debug,
               render,
               verbose,
               callbacks,
               seed, results_dir):

    if use_cuda:
        torch.set_num_threads(1)

    print('Env id: {}, Alg: Tree-MVD-PolicyGradient'.format(env_domain))

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
    actor_input_shape = mdp.info.observation_space.shape
    policy_params = dict(
        std_0=initial_std,
        n_features=n_features_actor,
        use_cuda=use_cuda
    )
    actor_optimizer = {
                       'class': optim.Adam,
                       'params': {'lr': lr_actor}}

    if squashed_policy:
        actor = GaussianTorchPolicyExtendedSquashed(ActorNetwork1,
                                                    actor_input_shape,
                                                    mdp.info.action_space.shape,
                                                    mdp.info.action_space.low,
                                                    mdp.info.action_space.high,
                                                    **policy_params)
    else:
        actor = GaussianTorchPolicyExtended(ActorNetwork1,
                                            actor_input_shape,
                                            mdp.info.action_space.shape,
                                            **policy_params)

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)

    approximator = ExtraTreesRegressor
    critic_params = dict(input_shape=critic_input_shape,
                         output_shape=(1,),
                         n_estimators=n_estimators_tree_fit,
                         criterion='mse',
                         max_depth=None,  # leave to None and specify min_samples_leaf instead
                         min_samples_split=min_samples_split_tree_fit,
                         min_samples_leaf=min_samples_leaf_tree_fit,
                         max_features=None,  # None := use all features. Default for regression tasks
                         bootstrap=True,
                         oob_score=False,
                         n_jobs=n_jobs_tree_fit,
                         verbose=False
                         )

    critic = Regressor(approximator, **critic_params)

    # Agent
    mc_gradient_estimator = {'estimator': mc_grad_estimator,
                             'n_samples': mc_samples_gradient,
                             'coupling': coupling
                             }
    agent = RegressionTrees_PolicyGradient(mdp.info, actor, actor_optimizer,
                                           critic,
                                           n_iters_critic=n_iters_critic,
                                           critic_fit_params=None,
                                           n_epochs_policy=n_epochs_policy, batch_size_policy=batch_size_policy,
                                           mc_samples_action_next=mc_samples_action_next,
                                           mc_gradient_estimator=mc_gradient_estimator,
                                           quiet=not verbose,
                                           use_replay_memory=use_replay_memory,
                                           max_replay_memory_size=max_replay_memory_size,
                                           replay_memory_samples=replay_memory_samples,
                                           )

    # Algorithm
    prepro = None
    if preprocess_states == 'minmax':
        prepro = MinMaxPreprocessor(mdp_info=mdp.info)
    elif preprocess_states == 'standard':
        prepro = StandardizationPreprocessor(mdp_info=mdp.info)

    plotter = None
    dataset_callback = None
    if debug:
        plotter = PlotDataset(mdp.info, obs_normalized=True if preprocess_states else False)

    visualization_callback = None
    if callbacks:
        agent._Q_approx.fit(np.random.random(size=(16, *critic_input_shape)).astype(NP_FLOAT_DTYPE),
                            np.random.random(16).astype(NP_FLOAT_DTYPE))
        vf = ValueFunction(agent)
        dataset_callback = CollectDataset()
        visualization_callback = visualization(env_domain, vf, results_dir, n_episodes_test, debug)

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
        env_domain='Pendulum-v0',
        env_task='NA',
        horizon=200,
        gamma=0.99,
        n_epochs=50,
        n_steps=3000,
        n_steps_per_fit=3000,
        n_episodes_test=10,
        n_iters_critic=10,
        mc_samples_action_next=1,
        n_estimators_tree_fit=50,
        min_samples_split_tree_fit=25,
        min_samples_leaf_tree_fit=2,
        n_jobs_tree_fit=1,
        use_replay_memory=False,
        max_replay_memory_size=500000,
        replay_memory_samples=1000,
        initial_std=1.0,
        squashed_policy=False,
        n_epochs_policy=5,
        batch_size_policy=32,
        n_features_actor=32,
        lr_actor=1e-4,
        mc_grad_estimator='mvd',
        mc_samples_gradient=1,
        coupling=False,
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

    parser.add_argument('--env-domain', type=str)
    parser.add_argument('--env-task', type=str)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--n-steps', type=int)
    parser.add_argument('--n-steps-per-fit', type=int)
    parser.add_argument('--n-episodes-test', type=int)
    parser.add_argument('--n-iters-critic', type=int)
    parser.add_argument('--mc-samples-action-next', type=int)
    parser.add_argument('--n-estimators-tree-fit', type=int)
    parser.add_argument('--min-samples-split-tree-fit', type=int)
    parser.add_argument('--min-samples-leaf-tree-fit', type=int)
    parser.add_argument('--n-jobs-tree-fit', type=int)
    parser.add_argument('--use-replay-memory', action='store_true')
    parser.add_argument('--max-replay-memory-size', type=int)
    parser.add_argument('--replay-memory-samples', type=int)
    parser.add_argument('--initial-std', type=float)
    parser.add_argument('--squashed-policy', action='store_true')
    parser.add_argument('--n-epochs-policy', type=int)
    parser.add_argument('--batch-size-policy', type=int)
    parser.add_argument('--n-features-actor', type=int)
    parser.add_argument('--lr-actor', type=float)
    parser.add_argument('--mc-grad-estimator', type=str)
    parser.add_argument('--mc-samples-gradient', type=int)
    parser.add_argument('--coupling', action='store_true')
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
