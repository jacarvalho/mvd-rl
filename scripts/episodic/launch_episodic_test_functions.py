import os
from tqdm import tqdm


from mushroom_rl.approximators.parametric import LinearApproximator
from mushroom_rl.approximators.regressor import Regressor
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J

from src.algs.episodic.black_box_optimization_mvd import EMVD
from src.algs.episodic.pgpe import PGPE
from src.algs.episodic.reptrick import RepTrick
from src.distributions.gaussians import GaussianDistributionDiagonalCovarianceLogParameterization
from src.optimizers.learning_rates import FixedLearningRate
from src.policies.return_weights_policy import ReturnWeightsPolicy
from src.envs.optim_test_functions import *
from src.utils.seeds import fix_random_seed
from src.utils.set_evaluation import SetEval

tqdm.monitor_interval = 0


def experiment(alg, params, test_function, n_epochs=1, n_eval_episodes=1, results_dir='/tmp/', seed=0):
    mdp = test_function['mdp']
    mu = np.copy(test_function['mu'])
    std = np.copy(test_function['std'])

    # Fix random seed
    fix_random_seed(seed, mdp)

    approximator = Regressor(LinearApproximator,
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape)

    policy = ReturnWeightsPolicy(mu=approximator)

    distribution = GaussianDistributionDiagonalCovarianceLogParameterization(mu, std)

    # Agent
    if alg.__name__ == 'RepTrick':
        agent = alg(mdp.info, distribution, policy, test_function['mdp'], **params)
    else:
        agent = alg(mdp.info, distribution, policy, **params)

    # Train
    means_l, covs_l = [], []
    n_evals_f_evals_l = []

    core = Core(agent, mdp)
    with SetEval(agent):
        dataset_eval = core.evaluate(n_episodes=n_eval_episodes, quiet=True)

    J = compute_J(dataset_eval, gamma=mdp.info.gamma)
    print('J at start : ' + str(np.mean(J)))

    means_l.append(distribution.mean)
    covs_l.append(distribution.cov)

    n_episodes_per_fit = 0
    if alg.__name__ == 'PGPE':
        n_episodes_per_fit = params['mc_samples_gradient']
    elif alg.__name__ == 'RepTrick':
        n_episodes_per_fit = params['mc_samples_gradient']
    elif alg.__name__ == 'EMVD':
        n_episodes_per_fit = params['mc_samples_gradient'] * distribution.theta_mvd_size

    for i in range(n_epochs):
        core.learn(n_episodes=n_episodes_per_fit,
                   n_episodes_per_fit=n_episodes_per_fit,
                   quiet=True
                   )

        if i % max(min(1000, int(0.1*n_epochs)), 1) == 0 or i == n_epochs - 1:
            distribution.print_dist_info()

            with SetEval(agent):
                dataset_eval = core.evaluate(n_episodes=n_eval_episodes, quiet=True)

            J = compute_J(dataset_eval, gamma=mdp.info.gamma)
            print('J at iteration ' + str(i) + ': ' + str(np.mean(J)))
            n_evals_f_evals_l.append((i, (i+1)*n_episodes_per_fit, np.mean(J)))

        means_l.append(distribution.mean)
        covs_l.append(distribution.cov)

    # Save results
    base_dir = os.path.join(results_dir, mdp.__name__, alg.__name__)
    os.makedirs(base_dir, exist_ok=True)
    np.save(os.path.join(base_dir, 'means.npy'), np.array(means_l))
    np.save(os.path.join(base_dir, 'covs.npy'), np.array(covs_l))
    np.save(os.path.join(base_dir, 'n_evals_f_evals.npy'), np.array(n_evals_f_evals_l))


test_functions = {
    'Quadratic': {
        'mdp': Quadratic(dim=2),
        'mu': -5. * np.ones(2),
        'std': 2.0 * np.ones(2),
        'lr': FixedLearningRate(value=5e-4)
        },
    'Himmelblau': {
        'mdp': Himmelblau(),
        'mu': np.array([0., -6.]),
        'std': 2.0 * np.ones(2),
        'lr': FixedLearningRate(value=5e-4)
    },
    'Styblinski': {
        'mdp': Styblinski(),
        'mu': np.array([0., 0.]),
        'std': 2.0 * np.ones(2),
        'lr': FixedLearningRate(value=5e-4)
    },
}

# MDP
test_functions_l = [
    'Quadratic',
    'Himmelblau',
    'Styblinski',
]

algs = [
        EMVD,
        PGPE,
        RepTrick,
        ]
params = [
          {'coupling': True, 'mc_samples_gradient': 1},
          {'baseline': True, 'mc_samples_gradient': 2*4},
          {'mc_samples_gradient': 2*4},
          ]

for alg, params in zip(algs, params):
    print('--------------------------------------')
    print('\n' + alg.__name__)
    for test_function in test_functions_l:
        print('\n' + test_function)
        params['grad_optimizer'] = test_functions[test_function]['lr']
        experiment(alg, params, test_functions[test_function],
                   n_epochs=1500,
                   n_eval_episodes=1,
                   results_dir=os.path.join('./logs/test-functions'),
                   seed=3)
