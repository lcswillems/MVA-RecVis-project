import torch as th

def concat_trajss(trajs1, trajs2):
    if trajs1 is None or trajs2 is None:
        return trajs1 or trajs2

    obss1, acts1 = trajs1
    obss2, acts2 = trajs2
    obss = th.cat((obss1, obss2))
    acts = th.cat((acts1, acts2))

    return obss, acts

def collect_perfect_trajs(env, expert, S, T):
    obss = th.zeros((S, T) + env.obs_shape)
    acts = th.zeros((S, T) + env.act_shape)

    for s in range(S):
        obs = env.reset()
        expert.reset(obs['cube_pos'], obs['goal_pos'])

        for t in range(T):
            obss[s, t] = obs
            acts[s, t] = expert.act(obs)

            obs, _, done, _ = env.step(acts[s, t])

            if done:
                break

    return obss, acts

def collect_corrected_trajs(env, learner, expert, β, S, T):
    obss = th.zeros((S, T) + env.obs_shape)
    acts = th.zeros((S, T) + env.act_shape)

    for s in range(S):
        obs = env.reset()
        expert.reset(obs['cube_pos'], obs['goal_pos'])

        for t in range(T):
            obss[s, t] = obs
            acts[s, t] = expert.act(obs)

            obs, _, done, _ = env.step(acts[s, t] if th.rand() < β else learner.act(obs))

            if done:
                break

    return obss, acts