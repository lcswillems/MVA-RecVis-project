import argparse
import gym
import numpy as np
from gym.utils import seeding

import utils

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=20,
                    help="number of iterations of DART (default: 20)")
parser.add_argument("--nb-demos", type=int, default=10,
                    help="number of demonstrations per iteration of DART (default: 10)")
parser.add_argument("--alpha", type=float, default=3,
                    help="alpha hyperparameter of DART (default: 3)")
parser.add_argument("--T", type=int, default=50,
                    help="time-horizon hyperparameter of DART (default: 50)")
parser.add_argument("--seed", type=int, default=1,
                    help="the seed (default: 1)")
args = parser.parse_args()

env = utils.make_env(args.seed)
expert = utils.PickPlaceExpert(env.dt)
learner = ???

def generate_demo(expert):
    demo = []

    obs = env.reset()
    expert.reset(obs['cube_pos'], obs['goal_pos'])

    while True:
        action = actor.act(obs)
        demo.append(obs, action)

        obs, _, done, _ = env.step(action)

        if done:
            break

    return demo

class GaussianExpert:
    def __init__(self, expert, Σ=None, np_random=np.random):
        self.expert = expert
        self.Σ = Σ
        self.np_random = np_random

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

    def act(self, obs):
        assert self.Σ is not None
        return self.np_random.multivariate_normal(self.expert.act(obs), self.Σ)

def DART(learner, expert):
    demos = [generate_demo(expert)]
    all_demos = demos
    gaussian_expert = GaussianExpert(expert)

    for k in range(args.iter):
        Σh = 1/args.T * np.sum([aaT(learner.act(o) - a) for o, a in d for d in demos], axis=0)
        Σ = args.alpha/(args.T * np.trace(Σh))

        gaussian_expert.Σ = Σ

        demos = [generate_demo(gaussian_expert) for n in range(args.nb_demos)]
        all_demos += demos

        learner.train(all_demos)