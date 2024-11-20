import socket
from experiment_launcher import Launcher
import torch

hostname = socket.gethostname()
LOCAL = False if hostname == 'mn01' or 'hla' in hostname else True

# Fix number of torch threads
if LOCAL:
    torch.set_num_threads(1)


local = LOCAL
test = False
use_cuda = False

launcher = Launcher(exp_name='off_policy_sac_extra_samples',
                    python_file='exp_sac',
                    # project_name='project01263',
                    n_exp=25,
                    n_cores=1,
                    memory=5000,
                    days=1,
                    hours=12,
                    minutes=59,
                    seconds=59,
                    n_jobs=1,
                    conda_env='ps-mvd',
                    gres='gpu:rtx2080:1' if use_cuda else None,
                    use_timestamp=True)

envs = ['InvertedPendulumBulletEnv-v0', 'Pendulum-v0', 'AntBulletEnv-v0',
        'HalfCheetahBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HopperBulletEnv-v0',
        'ReacherBulletEnv-v0', 'InvertedPendulumSwingupBulletEnv-v0']

horizons = [1000, 200, 1000, 1000, 1000, 1000, 1000, 1000]
gammas = 0.99
n_epochss = [100, 100, 100, 100, 100, 100, 100, 100]
n_stepss = [1000, 1000, 10000, 10000, 10000, 10000, 1000, 1000]
n_episodes_test = 10
initial_replay_sizes = [128, 128, 5000, 5000, 5000, 5000, 128, 128]
batch_sizes = [64, 64, 256, 256, 256, 256, 64, 64]
warmup_transitionss = [128, 128, 10000, 10000, 10000, 10000, 128, 128]
n_features_actors = [64, 64, 256, 256, 256, 256, 64, 64]
n_features_critics = [64, 64, 256, 256, 256, 256, 64, 64]
max_replay_size = 500000
lr_actor = 1e-4
lr_critic = 3e-4
preprocess_states = False
mc_samples_gradients = [2*2*1, 2*2*1, 2*2*8,
                        2*2*6, 2*2*6, 2*2*3,
                        2*2*2, 2*2*1]
verbose = LOCAL

for env, horizon, batch_size, n_epochs, n_steps, warmup_transitions, n_features_actor, n_features_critic, initial_replay_size, mc_samples_gradient in \
        zip(envs, horizons, batch_sizes, n_epochss, n_stepss, warmup_transitionss, n_features_actors, n_features_critics, initial_replay_sizes, mc_samples_gradients):

    launcher.add_default_params(horizon=horizon,
                                n_epochs=n_epochs,
                                n_steps=n_steps,
                                n_episodes_test=n_episodes_test,
                                initial_replay_size=initial_replay_size,
                                max_replay_size=max_replay_size,
                                batch_size=batch_size,
                                warmup_transitions=warmup_transitions,
                                n_features_actor=n_features_actor,
                                n_features_critic=n_features_critic,
                                lr_actor=lr_actor,
                                lr_critic=lr_critic,
                                mc_samples_gradient=mc_samples_gradient,
                                preprocess_states=preprocess_states,
                                use_cuda=use_cuda,
                                verbose=verbose)

    launcher.add_experiment(env_id=env)

    launcher.run(local, test)
