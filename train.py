# TODO: TrajectoriesManager :
#           - faire retourner les trajectoires récoltées par tm.collect_perfect_trajectories
#             et tm.collect_corrected_trajectories
#           - ne pas stocker les trajectoires dans le dataset pour
#             `val_trajs = tm.collect_corrected_trajs(S, None, expert, 1, T)` de DAgger
# TODO: finir d'adapter DART et DAgger
# TODO: mettre à jour les paramètres : mettre bonnes valeurs par déf, enlever des requirements

import argparse
import numpy as np
import os
import datetime

from utils import aaT, PickPlaceExpert, GaussianExpert, TrajectoriesManager, Learner

parser = argparse.ArgumentParser()
parser.add_argument("--algo", required=True)
parser.add_argument("--N", type=int, default=50)
parser.add_argument("--S", type=int, default=10)
parser.add_argument("--T", type=int, default=50)
parser.add_argument("--workers", type=int, default=16)
parser.add_argument("--model-basedir", default="storage/models")
parser.add_argument("--trajs-basedir", default="storage/trajs")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--alpha", type=float, default=3)
parser.add_argument("--p", type=float, default=.9)
args = parser.parse_args()

assert args.algo in ["bc", "dart", "dagger"]

# Seed numpy

np.random.seed(args.seed)

# Define trajs_dir and model_dir

suffix = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
slug = args.algo + "_" + suffix
trajs_dir = os.path.join(args.trajs_basedir, slug)
model_dir = os.path.join(args.model_basedir, slug)

# Instantiate trajectories manager, learner, expert

tm = TrajectoriesManager(trajs_dir, args.workers)
learner = Learner(model_dir, trajs_dir)
expert = PickPlaceExpert()

# The algorithms

if args.algo == "bc":

    for _ in range(args.N):
        tm.collect_perfect_trajs(args.S, expert, args.T)
        learner.train()

elif args.algo == "dart":

    gaussian_expert = GaussianExpert(expert)

    trajs_i = tm.collect_perfect_trajs(1, expert, args.T)

    for _ in range(args.N):
        Σh = 1/args.T * np.sum([aaT(learner.act(o) - a) for o, a in zip(d) for d in zip(trajs_i)], axis=0)
        Σ = args.alpha/(args.T * np.trace(Σh))
        gaussian_expert.Σ = Σ

        trajs_i = tm.collect_perfect_trajs(args.S, expert, args.T)
        learner.train()

elif args.algo == "dagger":

    learners = [gen_learner() for _ in range(args.N)]

    for i in range(args.N):
        β = args.p ** i
        learner = None if i == 0 else learners[i-1]
        tm.collect_corrected_trajs(args.S, learner, expert, β, args.T)
        learners[i].train()

    val_trajs = tm.collect_corrected_trajs(args.S, None, expert, 1, args.T)
    scores = np.array([learner.evaluate(val_trajs) for learner in learners])

    best_learner = learners[scores.argmax()]