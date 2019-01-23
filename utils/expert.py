import numpy as np
from utils import va, unva

class PickPlaceExpert:
    def reset(self, dt, cube_pos, goal_pos):
        self.dt = dt/4

    def act(self, obs):
        gripper_pos = obs['gripper_pos']
        cube_pos = obs['cube_pos']
        goal_pos = obs['goal_pos']
        gripper_state = obs['gripper_state']

        pos_threshold = 1e-2
        state_closed = .03
        state_opened = .05

        def move(dest, open):
            grip_velocity = int(open) * 2 - 1
            dist = dest - gripper_pos
            if (open and gripper_state > state_opened) or (not open and gripper_state < state_closed):
                max_v = 3
            else:
                max_v = 0

            v = dist / self.dt
            if np.linalg.norm(v) > max_v:
                v = v / np.linalg.norm(v) * max_v
            return dict(linear_velocity=v, grip_velocity=grip_velocity)

        if  np.linalg.norm((gripper_pos - cube_pos)[:2]) > pos_threshold :
            action = move(cube_pos + [0, 0, .1], True)
        elif np.linalg.norm(gripper_pos - cube_pos) > pos_threshold:
            action = move(cube_pos, True)
        elif gripper_state > state_closed:
            action = move(cube_pos, False)
        else:
            action = move(goal_pos, False)

        return action, action

class DAggerExpert:
    def __init__(self, expert, net):
        self.expert = expert
        self.net = net
        self.β = 1

    def reset(self, *args, **kwargs):
        self.expert.reset(*args, **kwargs)

    def act(self, obs):
        perfect_act, act = self.expert.act(obs)
        if np.random.rand() >= self.β:
            act = self.net.get_dic_action(obs)
        return perfect_act, act

class GaussianExpert:
    def __init__(self, expert, Σ=None):
        self.expert = expert
        self.Σ = Σ

    def reset(self, *args, **kwargs):
        return self.expert.reset(*args, **kwargs)

    def act(self, obs):
        perfect_act, _ = self.expert.act(obs)
        if self.Σ is None:
            return perfect_act, perfect_act
        return perfect_act, unva(np.random.multivariate_normal(va(perfect_act), self.Σ))