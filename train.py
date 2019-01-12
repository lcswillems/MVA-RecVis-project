# TODO: TrajectoriesManager :
#           - faire retourner les trajectoires récoltées par tm.collect_perfect_trajectories
#             et tm.collect_corrected_trajectories
#           - ne pas stocker les trajectoires dans le dataset pour
#             `val_trajs = tm.collect_corrected_trajs(S, None, expert, 1, T)` de DAgger
# TODO: finir d'adapter DART et DAgger

import numpy as np
import os
import datetime
from sacred import Experiment
from bc.config import model_ingredient, train_ingredient
import torch as th

from utils import aaT, va, transform_frames, PickPlaceExpert, GaussianExpert, TrajectoriesManager, Learner

ex = Experiment('train', ingredients=[train_ingredient])

@ex.config
def config():
    algo = None
    N = 10
    S = 2
    T = 50
    nb_workers = 2
    model_basedir = "storage/models"
    trajs_basedir = "storage/trajs"
    seed = 1
    α = 3
    p = .9

@ex.automain
def main(algo, N, S, T, nb_workers, model_basedir, trajs_basedir, seed, α, p, train):
    assert algo in ["bc", "dart", "dagger"]

    # Seed numpy

    np.random.seed(seed)

    # Define trajs_dir and model_dir

    suffix = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    slug = algo + "_" + suffix
    trajs_dir = os.path.join(trajs_basedir, slug)
    model_dir = os.path.join(model_basedir, slug)

    # Instantiate trajectories manager, learner, expert

    tm = TrajectoriesManager(trajs_dir, nb_workers)
    learner = Learner(model_dir, trajs_dir, train)
    expert = PickPlaceExpert()

    # The algorithms

    if algo == "bc":

        for _ in range(N):
            tm.collect_perfect_trajs(S, expert, T)
            learner.train()

    elif algo == "dart":

        gaussian_expert = GaussianExpert(expert)

        trajs_i = tm.collect_corrected_trajs(S, gaussian_expert, expert, 0, T)

        for _ in range(N):
            learner.train()
            with th.no_grad():
                mat = th.zeros((4, 4)).cuda()
                for t in trajs_i:
                    obs, act = t
                    obs = transform_frames(obs)
                    act = th.tensor(list(map(va, act)), dtype=th.float32).cuda()
                    obs = th.cat((
                        th.cat((th.zeros((2, 4, 224, 224)), obs[2:]), dim=0),
                        th.cat((th.zeros((1, 4, 224, 224)), obs[1:]), dim=0),
                        obs),
                        dim=1
                    )
                    pred = learner.act(obs)
                    pred_act = th.cat((
                            2 * (pred[:, 0] > pred[:, 1]).unsqueeze(1).float() - 1,
                            pred[:, 2:5],
                    ), dim=1)
                    diff = pred_act - act
                    mat += (diff[:, :, None] * diff[:, None, :]).sum(dim=0)

                Σh = 1/(T * S) * mat
                Σ = α * Σh.cpu().numpy() / (T * S * th.trace(Σh).item())
                gaussian_expert.Σ = Σ

            trajs_i = tm.collect_corrected_trajs(S, gaussian_expert, expert, 0, T)

    elif algo == "dagger":

        learners = [gen_learner() for _ in range(N)]

        for i in range(N):
            β = p ** i
            learner = None if i == 0 else learners[i-1]
            tm.collect_corrected_trajs(S, learner, expert, β, T)
            learners[i].train()

        val_trajs = tm.collect_perfect_trajs(S, expert, T)
        scores = np.array([learner.evaluate(val_trajs) for learner in learners])

        best_learner = learners[scores.argmax()]