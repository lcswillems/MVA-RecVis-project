import numpy as np
from pyquaternion import Quaternion
from gym.utils import seeding

# Slightly modified version of https://gitlab.inria.fr/rstrudel/bcmuj/blob/master/bcmuj/scene/camera.py

# all angles are defined using radians

class Camera:
    def __init__(self, radius=1, theta=0, phi=0, dr=0, dtheta=(0, 0), dphi=(0, 0), sphere_center=[0.5, 0.0, 0.5], np_random=np.random):
        self.radius = radius
        self.theta = theta
        self.phi = phi
        self.dr = dr
        self.dtheta = dtheta
        self.dphi = dphi
        self.sphere_center = sphere_center
        self.np_random = np_random

        q0 = Quaternion([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
        q1 = Quaternion([np.cos(np.pi/4), 0, np.sin(np.pi/4), 0])
        self.default_ref = q1*q0

        self._set_random_pose()

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        self._set_random_pose()

    def _set_random_pose(self):
        dtheta = self.dtheta
        dphi = self.dphi

        radius = self.radius+self.np_random.uniform(-0.1, 0.1)
        theta = self.theta+self.np_random.uniform(dtheta[0], dtheta[1])
        phi = self.phi+self.np_random.uniform(dphi[0], dphi[1])

        self._set_absolute_pose(radius, theta, phi)

    def _set_absolute_pose(self, radius, theta, phi):
        u = np.array([np.cos(phi/2), 0., -np.sin(phi/2), 0.])
        v = np.array([np.cos(theta/2), 0, 0., np.sin(theta/2)])

        cam_pos = self.sphere_center+self._sph2pos(radius, theta, phi)
        self.pos = cam_pos.copy()

        q = Quaternion(v)*Quaternion(u)*self.default_ref
        self.quat = self._q2arr(q).copy()

    def _sph2pos(self, r, theta, phi):
        x = r*np.cos(phi)*np.cos(theta)
        y = r*np.cos(phi)*np.sin(theta)
        z = r*np.sin(phi)
        return np.array([x, y, z])

    def _q2arr(self, q):
        return np.array([q[i] for i in range(4)])