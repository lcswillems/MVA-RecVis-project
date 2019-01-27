import numpy as np
import torch as th
import torch.utils.data as data

def va(action):
    """Vectorize the dict action."""

    return np.insert(action['linear_velocity'], 0, action['grip_velocity'])

def unva(action):
    return dict(linear_velocity=action[1:], grip_velocity=action[0])

def aaT(a):
    """Take a numpy vector a and compute the matrix a^T a."""

    a = a.reshape(-1, 1)
    return a.dot(a.T)

def transform_frames(frames):
    frames = th.stack([
        th.cat((
            th.tensor(d['rgb0'].copy()).cuda().permute(2, 0, 1).float() / 255,
            th.tensor(d['depth0'].copy()).cuda().unsqueeze(0).float() / 255
        ))
        for d in frames
    ])
    return (frames - .5) * 2

def transform_acts(acts):
    acts = th.stack([
        th.tensor(np.insert(a['linear_velocity'], 0, a['grip_velocity'])).float()
        for a in acts
    ])
    return acts

class MultiFrameDataset(data.Dataset):

    def __init__(self, tensors, indices, bound):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.indices = indices
        self.bound = bound

    def __getitem__(self, item):
        return tuple(
            th.cat([
                tensor[np.clip(item + idx, self.bound[item, 0].item(), self.bound[item, 1].item() - 1)]
                for idx in indices
            ], dim=0)
            for tensor, indices in zip(self.tensors, self.indices)
        )

    def __len__(self):
        return self.tensors[0].size(0)

def pred_to_act(pred):
    action = dict(
        grip_velocity=2 * (pred[0] > pred[1]) - 1,
        linear_velocity=pred[2:5]
    )
    return action

def pred_to_vector_act(pred):
    return np.concatenate((
        2 * (pred[:, 0:1] > pred[:, 1:2]) - 1,
        pred[:, 2:5],
    ), axis=1)