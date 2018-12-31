from utils import collect_perfect_trajs, concat_trajss

def BC(env, gen_learner, expert, N, S, T):
    learner = gen_learner()

    trajs = None

    for _ in range(N):
        trajs_i = collect_perfect_trajs(env, expert, S, T)
        trajs = concat_trajss(trajs, trajs_i)

        learner.train(trajs)