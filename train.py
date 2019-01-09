import numpy as np
import torch as th

from utils import aaT, GaussianExpert, TrajectoriesManager

# TODO: lire les arguments depuis quelque part (ligne de commande ou fichier)
# TODO: TrajectoriesManager :
#           - faire retourner les trajectoires récoltées par tm.collect_perfect_trajectories
#             et tm.collect_corrected_trajectories
#           - ne pas stocker les trajectoires dans le dataset pour
#             `val_trajs = tm.collect_corrected_trajs(S, None, expert, 1, T)` de DAgger
# TODO: implémenter une classe Learner qui permet d'entrainer le réseau à
#       partir du dataset_path en utilisant le code du prof. Je pense qu'il faut utiliser
#       learner = NetAgent(archi=archi, channels='rgbd', path=net_path,
#                          num_frames=3, skip=1, max_steps=-1,
#                          steps_action=4)
#       et regarder le script train de BC

tm = TrajectoriesManager(seed_init, dataset_path, nb_workers)
learner = Learner(dataset_path)

def BC(env, gen_learner, expert, N, S, T):
    learner = gen_learner()

    for _ in range(N):
        tm.collect_perfect_trajs(S, expert, T)
        learner.train()

def DART(env, gen_learner, expert, α, N, S, T, seed=None):
    learner = gen_learner()
    gaussian_expert = GaussianExpert(expert)
    gaussian_expert.seed(seed)

    trajs_i = tm.collect_perfect_trajs(1, expert, T)

    for _ in range(N):
        Σh = 1/T * np.sum([aaT(learner.act(o) - a) for o, a in zip(d) for d in zip(trajs_i)], axis=0)
        Σ = α/(T * np.trace(Σh))
        gaussian_expert.Σ = Σ

        trajs_i = tm.collect_perfect_trajs(S, expert, T)
        learner.train()

def DAgger(env, gen_learner, expert, p=.9, N=50, S=30, T=50):
    learners = [gen_learner() for _ in range(N)]

    for i in range(N):
        β = p ** i
        learner = None if i == 0 else learners[i-1]
        tm.collect_corrected_trajs(S, learner, expert, β, T)
        learners[i].train()

    val_trajs = tm.collect_corrected_trajs(S, None, expert, 1, T)
    scores = th.tensor([learner.evaluate(val_trajs) for learner in learners])

    return learners[scores.argmax()]