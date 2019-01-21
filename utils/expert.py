import numpy as np
import itertools
from utils import va, unva

class PickPlaceExpert:
    def reset(self, dt, cube_pos, goal_pos):
        self._dt = dt/4
        self._script = [
            ('grip_open', None),
            ('move_tool', (cube_pos+[0, 0, 0.1], 'open')),
            ('move_tool', (cube_pos+[0, 0, 0.0], 'open')),
            ('grip_close', None),
            ('move_tool', (goal_pos, 'close')),
        ]
        self._idx_script = 0
        self._gen = itertools.chain([])
        self._acc = False
        self._it_acc = 0

    def act(self, obs):
        self.gripper_pos = obs['gripper_pos']
        action = next(self._gen, None)
        if action is None:
            if not self._acc:
                self._idx_script += 1
            if self._idx_script < len(self._script):
                skill = self._script[self._idx_script]
                self._gen = self._compute_skill(skill, self.gripper_pos)
                action = next(self._gen, None)
            else:
                action = None
        if action is None:
            action = dict(linear_velocity=np.zeros(3), grip_velocity=0)
        return action, action

    def _compute_skill(self, skill, gripper_pos):
        name = skill[0]
        attributes = skill[1]

        if name == 'grip_open':
            for _ in range(5):
                yield dict(linear_velocity=np.zeros(3), grip_velocity=1)
        elif name == 'grip_close':
            for _ in range(5):
                yield dict(linear_velocity=np.zeros(3), grip_velocity=-1)
        elif name == 'move_tool':
            max_v = 3
            dt = self._dt

            pos_target = attributes[0]
            grip_velocity = int(attributes[1] == 'open')*2-1
            dist = np.subtract(pos_target, gripper_pos)

            while np.linalg.norm(dist) > .01:
                v = dist / dt
                if np.linalg.norm(v) > max_v:
                    v = v / np.linalg.norm(v) * max_v
                yield dict(linear_velocity=v, grip_velocity=grip_velocity)
                dist = np.subtract(pos_target, self.gripper_pos)

class GaussianExpert:
    def __init__(self, expert, Σ=None):
        self.expert = expert
        self.Σ = Σ

    def reset(self, *args, **kwargs):
        return self.expert.reset(*args, **kwargs)

    def act(self, obs):
        perfect_action, _ = self.expert.act(obs)
        if self.Σ is None:
            return perfect_action, perfect_action
        return perfect_action, unva(np.random.multivariate_normal(va(perfect_action), self.Σ))