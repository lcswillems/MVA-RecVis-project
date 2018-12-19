import torch as th
import torch.nn as nn
import gym

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.criterion = nn.CrossEntropyLoss
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

def train(obs, act, policy):
    raise NotImplementedError

def validate(obs, act, policy):
    return policy.criterion(act, policy(obs)).mean()

def collect(env, expert, policy, β, T=50, S=30):
    obs = th.zeros((S, T + 1) + env.obs_shape)
    act = th.zeros((S, T) + env.act_shape)
    for s in range(S):
        obs[s, 0] = env.reset()
        for t in range(T):
            act[s, t] = expert(obs[s, t])
            obs[s, t+1], _, done, _ = env.step(act[s, t] if th.rand() < β else policy(obs[s, t]))
            if done:
                raise NotImplementedError
    return obs[:, :-1], act

def dagger(env, expert):
    N = 50
    p = .9
    obs = th.zeros((0, 0) + env.obs_shape)
    act = th.zeros((0, 0) + env.act_shape)
    policies = [None] * (N + 1)
    for i in range(N):
        β = p ** i
        obs_i, act_i = collect(env, expert, policies[i], β)
        obs = th.cat((obs, obs_i))
        act = th.cat((act, act_i))
        policies[i+1] = train(obs, act, Policy())
    policies = policies[1:]
    val_data = collect(env, expert, None, 1)
    scores = th.tensor(list(map(lambda π: validate(*val_data, π), policies)))
    return policies[scores.argmax()]