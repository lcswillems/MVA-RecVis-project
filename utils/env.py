import gym
import numpy as np

from .camera import Camera

def make_env(seed):
    env = gym.make("FetchPickAndPlace-v1")
    wrapped_env = WrappedEnv(env)
    wrapped_env.seed(seed)
    return wrapped_env

class WrappedEnv(gym.Env):
    def __init__(self, env, resolution=(224, 224)):
        self.env = env
        self.resolution = resolution

        self.sim = self.env.unwrapped.sim
        self.dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        self.camera = Camera(dr=(0, 0.1), dtheta=(-np.pi/3, np.pi/3), dphi=(np.pi/20, np.pi/6))

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def _wrap_obs(self, obs):
        rgb, depth = self.sim.render(self.resolution[0], self.resolution[1], depth=True,
                                camera_name='external_camera_0')
        rgb = rgb[::-1]
        depth = (depth[::-1]*255).astype(np.uint8)

        wrapped_obs = {
            'rgb0': rgb,
            'depth0': depth,
            'gripper_pos': self.sim.data.get_site_xpos('robot0:grip'),
            'cube_pos': self.sim.data.get_site_xpos('object0'),
            'goal_pos': obs['desired_goal']
        }

        return wrapped_obs

    def _wrap_action(self, action):
        dx = np.zeros(4)
        dx[:3] = action['linear_velocity']/4
        dx[3] = action['grip_velocity']/2
        return dx

    def reset(self):
        obs = self.env.reset()

        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.env.unwrapped.goal - sites_offset[0]

        self.sim.model.cam_pos[-1] = self.camera.pos
        self.sim.model.cam_quat[-1] = self.camera.quat

        self.sim.forward()

        return self._wrap_obs(obs)

    def step(self, action):
        obs, r, done, info = self.env.step(self._wrap_action(action))
        return self._wrap_obs(obs), r, done, info

    def seed(self, seed=None):
        self.env.seed(seed)
        self.camera.seed(seed)