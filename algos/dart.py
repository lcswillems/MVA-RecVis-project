import numpy as np
from gym.utils import seeding

from utils import collect_perfect_trajs, concat_trajss, aaT

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

def DART(env, gen_learner, expert, α, N, S, T, seed=None):
    learner = gen_learner()
    gaussian_expert = GaussianExpert(expert)
    gaussian_expert.seed(seed)

    trajs_i = collect_perfect_trajs(env, expert, 1, T)
    trajs = trajs_i

    for _ in range(N):
        Σh = 1/T * np.sum([aaT(learner.act(o) - a) for o, a in zip(d) for d in zip(trajs_i)], axis=0)
        Σ = α/(T * np.trace(Σh))
        gaussian_expert.Σ = Σ

        trajs_i = collect_perfect_trajs(env, expert, S, T)
        trajs = concat_trajss(trajs, trajs_i)

        learner.train(trajs)