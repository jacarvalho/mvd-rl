import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mushroom_rl.core import Core
from mushroom_rl.solvers.lqr import compute_lqr_feedback_gain, compute_lqr_V_gaussian_policy, \
    compute_lqr_Q_gaussian_policy, compute_lqr_V_gaussian_policy_gradient_K, \
    _compute_lqr_Q_gaussian_policy_additional_term, _compute_lqr_Q_matrix
from mushroom_rl.utils.callbacks import PlotDataset
from mushroom_rl.utils.dataset import compute_J
from sklearn.metrics.pairwise import cosine_similarity

from src.algs.step_based.oracle_pg import Oracle_LQR_PolicyGradient
from src.envs.lqr_2states1actions import LQR2states1actions
from src.envs.lqr_2states2actions import LQR2states2actions
from src.envs.lqr_4states4actions import LQR4states4actions
from src.envs.lqr_6states6actions import LQR6states6actions
from src.envs.lqr_base import LQR
from src.policies.torch_policies import GaussianTorchPolicyExtended
from src.utils.dtype_utils import TORCH_FLOAT_DTYPE
from src.utils.seeds import fix_random_seed


########################################################################################################################

class ActorNetwork1(nn.Module):
    # Linear policy for LQR
    def __init__(self, input_shape, output_shape, **kwargs):
        super(ActorNetwork1, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_output, bias=False)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, **kwargs):
        features1 = self._h1(state.float())
        a = features1

        return a


class LQR_V_function:

    def __init__(self, lqr, dim_s, dim_a):
        self._lqr = lqr
        self._dim_s = dim_s
        self._dim_a = dim_a
        self._K = None
        self._Sigma = None

    def set_params(self, K, Sigma):
        self._K = np.atleast_2d(K)
        self._Sigma = Sigma

    def __call__(self, s):
        v = compute_lqr_V_gaussian_policy(s, self._lqr, self._K, self._Sigma)
        return v


class LQR_Q_function:

    def __init__(self, lqr, dim_s, dim_a, noise_q_amp_factor=0., noise_q_freq=100.):
        self._lqr = lqr
        self._dim_s = dim_s
        self._dim_a = dim_a
        self._K = None
        self._Sigma = None

        self._noise_q_amp_factor = noise_q_amp_factor
        self._noise_q_freq = noise_q_freq

    def set_params(self, K, Sigma):
        self._K = np.atleast_2d(K)
        self._Sigma = Sigma

    def __call__(self, sa):
        s = sa[:, :self._dim_s]
        a = sa[:, self._dim_s:]
        q = compute_lqr_Q_gaussian_policy(s, a, self._lqr, self._K, self._Sigma)

        p = np.random.dirichlet(np.ones(a.shape[1]), size=a.shape[0])
        noise = np.cos(2 * np.pi * self._noise_q_freq * (p * a).sum(axis=1, keepdims=True) +
                       np.random.uniform(0, 2*np.pi, size=(a.shape[0], a.shape[1])).sum(axis=1, keepdims=True)
                       )

        noise = self._noise_q_amp_factor * q * noise
        return q + noise


class LQR_Q_function_Torch(LQR_Q_function):

    def __init__(self, lqr, dim_s, dim_a, noise_q_amp_factor=0., noise_q_freq=100., use_cuda=False):
        super(LQR_Q_function_Torch, self).__init__(lqr, dim_s, dim_a, noise_q_amp_factor, noise_q_freq)
        self._torch_device = torch.device('cuda' if use_cuda else 'cpu')

    def __call__(self, sa):
        s = sa[:, :self._dim_s]
        a = sa[:, self._dim_s:]
        q = self.compute_Q_gaussian_policy(s.detach(), a, self._lqr, self._K, self._Sigma)

        p = torch.tensor(np.random.dirichlet(np.ones(a.shape[1]), size=a.shape[0]), dtype=TORCH_FLOAT_DTYPE)
        noise = torch.cos(2 * np.pi * self._noise_q_freq * (p * a).sum(dim=1, keepdim=True) +
                          torch.FloatTensor(a.shape[0], a.shape[1]).uniform_(0, 2*np.pi).sum(dim=1, keepdim=True)
                          )
        noise = self._noise_q_amp_factor * q.detach() * noise
        return q + noise

    def compute_Q_gaussian_policy(self, s, a, lqr, K, Sigma):
        b = _compute_lqr_Q_gaussian_policy_additional_term(lqr, K, Sigma)
        b = torch.tensor(b, dtype=TORCH_FLOAT_DTYPE, device=self._torch_device)
        return self.compute_Q(s, a, lqr, K) - b

    def compute_Q(self, s, a, lqr, K):
        if s.ndim == 1:
            s = s.reshape((1, -1))
        if a.ndim == 1:
            a = a.reshape((1, -1))
        sa = torch.cat((s, a), dim=1)

        M = _compute_lqr_Q_matrix(lqr, K)
        M = torch.tensor(M, dtype=TORCH_FLOAT_DTYPE, device=self._torch_device)

        return -1. * torch.einsum('...k,kl,...l->...', sa, M, sa).reshape(-1, 1)


class LQR_Policy_Gradient:

    def __init__(self, lqr):
        self._lqr = lqr

    def __call__(self, policy):
        K = -1. * np.atleast_2d(policy.get_weights())
        K = K.reshape(policy._mu.output_shape[0], policy._mu.input_shape[0])
        Cov = np.atleast_2d(policy.get_cov().detach().clone().cpu().numpy())
        s0 = self._lqr._initial_state
        return compute_lqr_V_gaussian_policy_gradient_K(s0, self._lqr, K, Cov)


ENV_SPECS = {
             'LQR2states1actions': {'mdp': LQR2states1actions,
                                   'initial_weights': [0.12090783, -1.90293579]
                                   },
             'LQR2states2actions': {'mdp': LQR2states2actions,
                                    'initial_weights': [-1.76504018, 2.1788581,
                                                        7.49035866, -4.86692174]
                                    },
             'LQR4states4actions': {'mdp': LQR4states4actions,
                                    'initial_weights': [1.44065069, -0.18045467, -0.28257178, -3.05142737,
                                                        2.74188152, -0.18942766, 0.13157326, 1.51403868,
                                                        0.12377449, -3.152568, 0.71876384, 0.46573287,
                                                        -0.18589064, -0.71770191, 1.86779365, -2.6336564]
                                    },
             'LQR6states6actions': {'mdp': LQR6states6actions,
                                    'initial_weights': [-0.98619249, 0.45078575, -0.6909792, -0.48931221, -0.37679959, -1.14926572,
                                                        1.60308272, -0.336829, 0.01307296, 0.18106089, -0.51512445, -0.76282749,
                                                        0.49639032, -0.13147592, 0.64799724, 1.90951081, -0.04233188, -0.45617268,
                                                        -0.37838526, -0.09254851, -1.80335514, -1.18366594, 0.38366327, -2.08020995,
                                                        0.9078764, -0.21197676, 1.87803645, -0.25504618, -0.27380292, -0.51829845,
                                                        0.69610819, 0.53745563, -0.45593299, 0.78232605, 1.73990422, 1.80203499]
                                    }
             }


def experiment(env_domain,
               horizon, gamma,
               n_epochs,
               n_episodes_learn,
               n_episodes_test,
               initial_std,
               n_epochs_policy,
               lr_actor,
               mc_grad_estimator,
               mc_samples_gradient,
               n_actions_per_state,
               coupling,
               noise_q_amp_factor,
               noise_q_freq,
               noise_a_type,
               use_cuda,
               debug,
               render,
               verbose,
               seed, results_dir):

    # Fix seed
    fix_random_seed(seed)

    if use_cuda:
        torch.set_num_threads(1)

    print(f'Env id: {env_domain}, Alg: Oracle PG')

    # Create results directory
    results_dir = os.path.join(results_dir, str(seed))
    os.makedirs(results_dir, exist_ok=True)

    # MDP
    if 'lqr_' in env_domain:
        lqr_id = env_domain.split('_')[-1]
        A = np.load(os.path.join('lqrs', lqr_id, 'A.npy'), allow_pickle=True)
        B = np.load(os.path.join('lqrs', lqr_id, 'B.npy'), allow_pickle=True)
        Q = np.load(os.path.join('lqrs', lqr_id, 'Q.npy'), allow_pickle=True)
        R = np.load(os.path.join('lqrs', lqr_id, 'R.npy'), allow_pickle=True)
        K_init = np.load(os.path.join('lqrs', lqr_id, 'K_init.npy'), allow_pickle=True)
        mdp = LQR(A, B, Q, R, max_pos=np.inf, max_action=np.inf,
                 random_init=False, episodic=False, gamma=gamma, horizon=50,
                 initial_state=None)
    else:
        mdp = ENV_SPECS[env_domain]['mdp'](horizon=horizon, gamma=gamma)
    fix_random_seed(seed, mdp)

    s_dim = mdp.info.observation_space.shape[0]
    a_dim = mdp.info.action_space.shape[0]

    K_opt = compute_lqr_feedback_gain(mdp)
    s0 = mdp.reset()
    mdp._initial_state = s0
    V_opt = compute_lqr_V_gaussian_policy(s0, mdp,
                                          K_opt,
                                          np.atleast_2d(initial_std**2 * np.eye(mdp.info.action_space.shape[0])))

    print()
    print(f'K_opt: {K_opt}')
    print(f'V_K_opt: {V_opt}')

    # initial weights
    if 'lqr_' in env_domain:
        init_weights = K_init
    else:
        initial_weights = ENV_SPECS[env_domain]['initial_weights']
        init_weights = np.array(initial_weights).reshape((mdp.info.action_space.shape[0],
                                                          mdp.info.observation_space.shape[0]))

    eigvals, eigvects = np.linalg.eig(mdp.A)
    if np.all(np.absolute(eigvals) < 1):
        print("A matrix is STABLE")
    else:
        print("A matrix is UNSTABLE")

    K_init = np.atleast_2d(init_weights)
    eigvals, eigvects = np.linalg.eig(mdp.A - mdp.B @ K_init)
    print(f'eigvals(A-B@K): {eigvals}')
    assert np.all(np.absolute(eigvals) < 1), "(A-BK) is UNSTABLE"

    V_K_init = compute_lqr_V_gaussian_policy(
        mdp._initial_state, mdp, K_init, np.atleast_2d(initial_std**2 * np.eye(a_dim)))
    print()
    print(f'K: {K_init}')
    print(f'V_K: {V_K_init}')

    # Set number of steps
    n_steps = n_episodes_learn * horizon
    n_steps_per_fit = n_steps

    # Approximators
    actor_input_shape = mdp.info.observation_space.shape
    policy_params = dict(
        std_0=initial_std,
        trainable_std=False,
        use_cuda=use_cuda,
    )
    actor_optimizer = {
                       'class': optim.Adam,
                       'params': {'lr': lr_actor}}

    actor = GaussianTorchPolicyExtended(ActorNetwork1,
                                        actor_input_shape,
                                        mdp.info.action_space.shape,
                                        **policy_params)

    # set initial weigths
    actor.set_weights(-1. * np.atleast_2d(init_weights))

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)

    # Agent
    mc_gradient_estimator = {'estimator': mc_grad_estimator,
                             'n_samples': mc_samples_gradient,
                             'coupling': coupling
                             }

    v_function = LQR_V_function(mdp, actor_input_shape[0], mdp.info.action_space.shape[0])
    if mc_gradient_estimator['estimator'] == 'reptrick':
        q_function = LQR_Q_function_Torch(mdp, actor_input_shape[0], mdp.info.action_space.shape[0],
                                          noise_q_amp_factor=noise_q_amp_factor, noise_q_freq=noise_q_freq)
    else:
        q_function = LQR_Q_function(mdp, actor_input_shape[0], mdp.info.action_space.shape[0],
                                    noise_q_amp_factor=noise_q_amp_factor, noise_q_freq=noise_q_freq)

    true_gradient = LQR_Policy_Gradient(mdp)

    agent = Oracle_LQR_PolicyGradient(mdp.info, actor, actor_optimizer,
                                      q_function, v_function,
                                      n_episodes_learn=n_episodes_learn,
                                      n_epochs_policy=n_epochs_policy,
                                      mc_gradient_estimator=mc_gradient_estimator,
                                      true_gradient=true_gradient,
                                      quiet=not verbose,
                                      )

    # Algorithm
    prepro = None
    plotter = None
    dataset_callback = None
    if debug:
        plotter = PlotDataset(mdp.info, obs_normalized=True if prepro != None else False)

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
    print(f'J: {np.mean(Jgamma):.4f} - H(pi): {entropy:.6f}\n')
    Jgamma_l.append((0, np.mean(Jgamma)))
    R_l.append((0, np.mean(R)))
    entropy_l.append((0, entropy))

    for n in range(n_epochs):
        print()
        print(f'-> Epoch: {n}/{n_epochs-1}, #samples: {(n+1)*n_steps}')

        print(f"K: {-1. * agent.policy.get_weights()}")
        core.learn(n_steps=n_steps, n_steps_per_fit=n_steps_per_fit, quiet=not verbose)

        dataset = core.evaluate(n_episodes=n_episodes_test, render=True if n % 10 == 0 and render else False,
                                quiet=not verbose)
        Jgamma = compute_J(dataset, gamma)
        R = compute_J(dataset)
        entropy = agent.policy.entropy()
        print(f'J: {np.mean(Jgamma):.3f} - H(pi): {entropy:.3f}')
        Jgamma_l.append((n_steps*(n+1), np.mean(Jgamma)))
        R_l.append((n_steps*(n+1), np.mean(R)))
        entropy_l.append((n_steps*(n+1), entropy))

        # Save agent and results
        # if n % max(1, int(n_epochs * 0.05)) == 0 or n == n_epochs - 1:
        if n % 1 == 0 or n == n_epochs - 1:
            np.save(os.path.join(results_dir, 'Jgamma.npy'), np.array(Jgamma_l))
            np.save(os.path.join(results_dir, 'R.npy'), np.array(R_l))
            np.save(os.path.join(results_dir, 'entropy.npy'), np.array(entropy_l))

    true_grads, grads = agent.get_grads()
    print_grads(true_grads[0].squeeze(), grads[0].squeeze())
    params = agent.get_policy_params()
    np.save(os.path.join(results_dir, 'true_grads.npy'), np.array(true_grads))
    np.save(os.path.join(results_dir, 'grads.npy'), np.array(grads))
    np.save(os.path.join(results_dir, 'params.npy'), np.array(params))


def print_grads(true_grads, grads):
    print('GRADS')
    if true_grads.ndim == 0:
        true_grads = true_grads.reshape((1, ))
        grads = grads.reshape((1,))

    for tg, g in zip(true_grads, grads):
        print(f'{tg:8.2f} \t {g:8.2f}')

    print('NORM G/TG')
    print(f'{np.linalg.norm(grads)/np.linalg.norm(true_grads):10.2f}')
    print('COSINE')
    print(f'{cosine_similarity(grads.reshape(1, -1), true_grads.reshape(1, -1)).item()}')


def default_params():
    defaults = dict(
        env_domain='LQR1States1Actions',
        horizon=1000,
        gamma=0.99,
        n_epochs=1,
        n_episodes_learn=1,
        n_episodes_test=1,
        initial_std=0.1,
        n_epochs_policy=1,
        lr_actor=1e-2,
        mc_grad_estimator='mvd',
        mc_samples_gradient=1,
        n_actions_per_state=2,
        coupling=False,
        noise_q_amp_factor=0.0,
        noise_q_freq=100.,
        noise_a_type='random_p',
        use_cuda=False,
        debug=False,
        verbose=False,
        render=False,
        seed=2,
        results_dir='/tmp/results'
    )

    return defaults


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env-domain', type=str)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--n-epochs', type=int)
    parser.add_argument('--n-episodes-learn', type=int)
    parser.add_argument('--n-episodes-test', type=int)
    parser.add_argument('--initial-std', type=float)
    parser.add_argument('--n-epochs-policy', type=int)
    parser.add_argument('--lr-actor', type=float)
    parser.add_argument('--mc-grad-estimator', type=str)
    parser.add_argument('--mc-samples-gradient', type=int)
    parser.add_argument('--n-actions-per-state', type=int)
    parser.add_argument('--coupling', action='store_true')
    parser.add_argument('--noise-q-amp-factor', type=float)
    parser.add_argument('--noise-q-freq', type=float)
    parser.add_argument('--noise-a-type', type=str)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render', action='store_true')
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
