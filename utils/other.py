import numpy as np
import torchvision.transforms as transforms
import torch as th

def va(action):
    """Vectorize the dict action."""

    return np.append(action['linear_velocity'], action['grip_velocity'])

def unva(action):
    return dict(linear_velocity=action[:3], grip_velocity=action[3])

def aaT(a):
    """Take a numpy vector a and compute the matrix a^T a."""

    a = a.reshape(-1, 1)
    return a.dot(a.T)

def transform_frames(frames):
    frames = th.stack(list(map(
        lambda d: th.cat((
            th.tensor(d['rgb0']).permute(2, 0, 1).float() / 255,
            th.tensor(d['depth0']).unsqueeze(0).float() / 255
        )),
        frames
    )))
    return (frames - .5) * 2