import torch as th

from utils import collect_corrected_trajs, concat_trajss

def DAgger(env, gen_learner, expert, p=.9, N=50, S=30, T=50):
    learners = [gen_learner() for _ in range(N)]

    trajs = None

    for i in range(N):
        β = p ** i
        learner = None if i == 0 else learners[i-1]
        trajs_i = collect_corrected_trajs(env, learner, expert, β, S, T)
        trajs = concat_trajss(trajs, trajs_i)

        learners[i].train(trajs)

    val_trajs = collect_corrected_trajs(env, None, expert, 1, S, T)
    scores = th.tensor([learner.evaluate(val_trajs) for learner in learners])

    return learners[scores.argmax()]