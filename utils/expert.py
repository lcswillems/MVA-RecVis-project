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
        gripper_pos = obs['gripper_pos']
        action = next(self._gen, None)
        if action is None:
            if not self._acc:
                self._idx_script += 1
            if self._idx_script < len(self._script):
                skill = self._script[self._idx_script]
                self._gen = self._compute_skill(skill, gripper_pos)
                action = next(self._gen, None)
            else:
                action = None
        if action is None:
            action = dict(linear_velocity=np.zeros(3), grip_velocity=0)
        return action

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
            max_v = 1
            t_acc = 0.01
            dt = self._dt

            pos_target = attributes[0]
            grip_velocity = int(attributes[1] == 'open')*2-1
            dist = np.subtract(pos_target, gripper_pos)

            t_dec = np.linalg.norm(dist) / max_v
            t_acc = np.min([t_acc, t_dec])
            t_end = t_dec + t_acc
            v_max = dist / t_dec

            if not self._acc:
                self._it_acc = 0
                self._acc = True
                for t in np.arange(0.0, t_end, dt) + dt:
                    k = 1.0
                    if t > t_end:
                        k = 0.0
                    elif t <= t_acc:
                        k = t / t_acc
                    elif t >= t_dec:
                        break
                    yield dict(linear_velocity=v_max * k,
                               grip_velocity=grip_velocity)
            else:
                self._it_acc += 1
                for t in np.arange(0.0, t_end, dt) + dt:
                    k = 1.0
                    if t >= t_dec:
                        k = 1 - (t - t_dec) / t_acc
                    yield dict(linear_velocity=v_max * k,
                               grip_velocity=grip_velocity)
                dist = np.subtract(pos_target, gripper_pos)
                if np.linalg.norm(dist) < 0.02 or self._it_acc > 4:
                    self._acc = False

class GaussianExpert:
    def __init__(self, expert, Σ=None):
        self.expert = expert
        self.Σ = Σ

    def reset(self, *args, **kwargs):
        return self.expert.reset(*args, **kwargs)

    def act(self, obs):
        if self.Σ is None:
            return self.expert.act(obs)
        return unva(np.random.multivariate_normal(va(self.expert.act(obs)), self.Σ))