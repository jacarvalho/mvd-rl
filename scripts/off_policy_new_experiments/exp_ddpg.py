import os
import argparse
import json

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.core import Core
from mushroom_rl.environments.gym_env import Gym
from mushroom_rl.policy import OrnsteinUhlenbeckPolicy
from mushroom_rl.utils.callbacks import PlotDataset
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.preprocessors import StandardizationPreprocessor
from mushroom_rl.algorithms.actor_critic import DDPG

from src.utils.seeds import fix_random_seed


########################################################################################################################
class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, features, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        dim_action = kwargs['action_shape'][0]

        n_output = output_shape[0]

        # Assume there are two hidden layers
        assert len(features) == 2, 'DDPG critic needs 2 hidden layers'

        self._h1 = nn.Linear(dim_state, features[0])
        self._h2_s = nn.Linear(features[0], features[1])
        self._h2_a = nn.Linear(dim_action, features[1], bias=False)
        self._h3 = nn.Linear(features[1], n_output)

        fan_in_h1, _ = nn.init._calculate_fan_in_and_fan_out(self._h1.weight)
        nn.init.uniform_(self._h1.weight, a=-1 / np.sqrt(fan_in_h1), b=1 / np.sqrt(fan_in_h1))

        fan_in_h2_s, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_s.weight)
        nn.init.uniform_(self._h2_s.weight, a=-1 / np.sqrt(fan_in_h2_s), b=1 / np.sqrt(fan_in_h2_s))

        fan_in_h2_a, _ = nn.init._calculate_fan_in_and_fan_out(self._h2_a.weight)
        nn.init.uniform_(self._h2_a.weight, a=-1 / np.sqrt(fan_in_h2_a), b=1 / np.sqrt(fan_in_h2_a))

        nn.init.uniform_(self._h3.weight, a=-3e-3, b=3e-3)

    def forward(self, state, action):
        state = state.float()
        action = action.float()

        features1 = F.relu(self._h1(state))
        features2_s = self._h2_s(features1)
        features2_a = self._h2_a(action)
        features2 = F.relu(features2_s + features2_a)

        q = self._h3(features2)
        return torch.squeeze(q)


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()

        dim_state = input_shape[0]
        dim_action = output_shape[0]

        self._action_scaling = torch.tensor(kwargs['action_scaling']).to(
            device=torch.device('cuda' if kwargs['use_cuda'] else 'cpu'))

        # Assume there are two hidden layers
        features = kwargs['features']
        assert len(features) == 2, 'DDPG critic needs to hidden layers'

        self._h1 = nn.Linear(dim_state, features[0])
        self._h2 = nn.Linear(features[0], features[1])
        self._h3 = nn.Linear(features[1], dim_action)

        fan_in_h1, _ = nn.init._calculate_fan_in_and_fan_out(self._h1.weight)
        nn.init.uniform_(self._h1.weight, a=-1 / np.sqrt(fan_in_h1), b=1 / np.sqrt(fan_in_h1))

        fan_in_h2, _ = nn.init._calculate_fan_in_and_fan_out(self._h2.weight)
        nn.init.uniform_(self._h2.weight, a=-1 / np.sqrt(fan_in_h2), b=1 / np.sqrt(fan_in_h2))

        nn.init.uniform_(self._h3.weight, a=-3e-3, b=3e-3)

    def forward(self, state):
        state = state.float()

        features1 = F.relu(self._h1(state))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        a = self._action_scaling * torch.tanh(a)
        return a



def experiment(env_id, horizon, gamma,
               n_epochs, n_steps, n_episodes_test,
               initial_replay_size,
               max_replay_size,
               batch_size,
               tau,
               features,
               lr_actor,
               lr_critic,
               preprocess_states,
               use_cuda,
               debug,
               verbose,
               seed, results_dir):

    if use_cuda:
        torch.set_num_threads(1)

    print('Env id: {}, Alg: DDPG'.format(env_id))

    # Create results directory
    results_dir = os.path.join(results_dir, str(seed))
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    mdp = Gym(env_id, horizon, gamma)

    # Fix seed
    fix_random_seed(seed, mdp)

    # Policy
    policy_class = OrnsteinUhlenbeckPolicy
    policy_params = dict(sigma=np.ones(1) * .2, theta=.15, dt=1e-2)

    # Approximator
    features = [int(elem) for elem in features.split("-")]
    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=ActorNetwork,
                        features=features,
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape,
                        use_cuda=use_cuda,
                        action_scaling=np.abs(mdp.info.action_space.high))

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': lr_actor}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': lr_critic, 'weight_decay': 1e-2}},
                         loss=F.mse_loss,
                         features=features,
                         input_shape=mdp.info.observation_space.shape,
                         action_shape=mdp.info.action_space.shape,
                         output_shape=(1,),
                         use_cuda=use_cuda)

    # Agent
    agent = DDPG(mdp.info, policy_class, policy_params,
                actor_params, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, tau)


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
    core.learn(n_steps=initial_replay_size, n_steps_per_fit=initial_replay_size, quiet=False if debug else True)

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
        env_id='Hopper-v3',
        horizon=1000, gamma=0.99,
        n_epochs=2, n_steps=5000, n_episodes_test=10,
        initial_replay_size=10000,
        max_replay_size=1000000,
        batch_size=64,
        tau=0.001,
        features="400-300",
        lr_actor=1e-4,
        lr_critic=1e-3,
        preprocess_states=False,
        use_cuda=False,
        debug=False,
        verbose=False,
        seed=1, results_dir='/tmp/results'
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-id', type=str, default='Hopper-v3')
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--n-steps', type=int)
    parser.add_argument('--n-episodes-test', type=int)
    parser.add_argument('--initial-replay-size', type=int)
    parser.add_argument('--max-replay-size', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--tau', type=float)
    parser.add_argument('--features', type=str)
    parser.add_argument('--lr-actor', type=float)
    parser.add_argument('--lr-critic', type=float)
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

