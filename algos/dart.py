import argparse
import gym
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="FetchPickAndPlace-v1",
                    help="name of the environment (default: FetchPickAndPlace-v1)")
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

# Create environment

env = gym.make(args.env)

# Seed numpy and environment

env.seed(args.seed)
np.random.seed(seed)

# Instantiate the learner

learner = ???

# Instantiate the supervisor

supervisor = ???

# Define gaussian supervisor

class GaussianSupervisor(Supervisor):
    def __init__(self, supervisor, Σ):
        self.supervisor = supervisor
        self.Σ = Σ

    def act(self, obs):
        return np.random.multivariate_normal(self.supervisor.act(obs), self.Σ)

# Generate a demonstration from an actor

def generate_demo(actor):
    demo = []

    obs = env.reset()
    while True:
        action = actor.act(obs)
        demo.append(obs, action)

        obs, _, done, _ = env.step(action)

        if done:
            break

    return demo

# DART algorithm

def compute_Σ(demos):
    Σh = 1/args.T * np.sum([aaT(learner.act(o) - a) for o, a in d for d in demos], axis=0)
    Σ = args.alpha/(args.T * np.trace(Σh))
    return Σ

demos = [generate_demo(supervisor)]
all_demos = demos

for k in range(args.iter):
    Σ = compute_Σ(demos)
    gaussian_supervisor = GaussianSupervisor(supervisor, Σ)

    demos = [generate_demo(gaussian_supervisor) for n in range(args.nb_demos)]
    all_demos += demos

    learner.train(all_demos)

learner.train(demos)