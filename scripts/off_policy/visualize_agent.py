import json
import os
import sys

from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils import VideoRecorder
from src.algs.step_based.sac_pg import SAC_PolicyGradient



########################################################################################################################
EXPERIMENT_DIR = '/tmp/results/1'

args = json.load(open(os.path.join(EXPERIMENT_DIR, 'args.json'), 'r'))

mdp = Gym(args['env_id'], args['horizon'], args['gamma'])


agent = SAC_PolicyGradient.load(os.path.join(EXPERIMENT_DIR, 'agent.msh'))


record_dict = dict(
    recorder_class=VideoRecorder,
    path=EXPERIMENT_DIR,
    tag='SAC-MVD',
    video_name="recording",
)
core = Core(agent, mdp, record_dictionary=record_dict)

core.evaluate(n_episodes=1000, render=True, record=True)

