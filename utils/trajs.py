import os
import pickle as pkl
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import copy
from utils import transform_frames
import torch as th

from bc.dataset.dataset_lmdb import DatasetWriter
from bc.dataset.utils import compress_images, process_trajectory, gather_dataset

from . import make_env

class TrajectoriesManager:
    def __init__(self, trajs_dir, nb_workers=1):
        self.trajs_dir = trajs_dir
        self.nb_workers = nb_workers

        self.seed = 1

    def _store_traj(self, obss, actions, seed):
        frames_chunk, scalars_chunk = process_trajectory(
            obss, actions, range(len(obss)),
            seed, jpeg_compression=False)

        path_worker_dataset = os.path.join(self.trajs_dir, '{:06}'.format(seed))

        os.makedirs(path_worker_dataset, exist_ok=True)
        dataset = DatasetWriter(path_worker_dataset, '', rewrite=True, float_depth=False)
        dataset.init_db()
        dataset.write_frames(frames_chunk)
        dataset.close_db()

        scalars = {}
        scalars.update(scalars_chunk)
        pkl.dump(scalars, open(path_worker_dataset + '.pkl', 'wb'))

    def collect_trajs(self, nb_trajs, traj_collector, *params):
        def traj_collector_wrapper(seed, params):
            env = make_env(seed)
            obss, compressed_obss, actions = traj_collector(env, *params)
            self._store_traj(compressed_obss, copy.deepcopy(actions), seed)
            return obss, actions

        print('Collecting trajectories {}-{}'.format(self.seed, self.seed + nb_trajs - 1))
        trajs = Parallel(n_jobs=self.nb_workers)(
            delayed(traj_collector_wrapper)(seed, params)
            for seed in tqdm(range(self.seed, self.seed + nb_trajs)))
        self.seed += nb_trajs

        print('Gathering trajectories dataset into one dataset...')
        gather_dataset(self.trajs_dir)

        return trajs

    def collect_perfect_trajs(self, nb_trajs, expert, T):
        return self.collect_trajs(nb_trajs, collect_perfect_traj, expert, T)

    def collect_corrected_trajs(self, nb_trajs, learner, expert, β, T):
        return self.collect_trajs(nb_trajs, collect_corrected_traj, learner, expert, β, T)

def collect_perfect_traj(env, expert, T):
    return collect_corrected_traj(env, None, expert, 1, T)

def collect_corrected_traj(env, learner, expert, β, T):
    obss = []
    compressed_obss = []
    actions = []

    obs = env.reset()
    expert.reset(env.dt, obs['cube_pos'], obs['goal_pos'])

    for _ in range(T):
        perfect_action, action = expert.act(obs)
        if np.random.rand() >= β:
            with th.no_grad():
                action = learner.act(th.cat((
                    th.tensor(obs['rgb0'].copy()).permute(2, 0, 1).float() / 255,
                    th.tensor(obs['depth0'].copy()).unsqueeze(0).float() / 255
                )).unsqueeze(0)).view(-1)
                action = dict(
                    grip_velocity=2 * (action[0] > action[1]).item(),
                    linear_velocity=action[2:5].cpu().numpy(),
                )

        obss.append(obs)
        compressed_obs = copy.deepcopy(obs)
        compress_images(compressed_obs)
        compressed_obss.append(compressed_obs)
        actions.append(perfect_action)

        obs, _, done, _ = env.step(action)

        if done:
            break

    return obss, compressed_obss, actions