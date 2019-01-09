import os
import pickle as pkl
from joblib import Parallel, delayed
from tqdm import tqdm
import torch as th

from bc.dataset.dataset_lmdb import DatasetWriter
from bc.dataset.utils import compress_images, process_trajectory, gather_dataset

from . import make_env

class TrajectoriesManager:
    def __init__(self, seed_init, dataset_path, nb_workers=1):
        self.seed = seed_init
        self.dataset_path = dataset_path
        self.nb_workers = nb_workers

    def _store_traj(self, obss, actions, seed):
        frames_chunk, scalars_chunk = process_trajectory(
            obss, actions, range(len(obss)),
            seed, jpeg_compression=False)

        path_worker_dataset = os.path.join(self.dataset_path, '{:06}'.format(seed))

        dataset = DatasetWriter(path_worker_dataset, '', rewrite=True, float_depth=False)
        dataset.init_db()
        dataset.write_frames(frames_chunk)
        dataset.close_db()

        scalars = {}
        scalars.update(scalars_chunk)
        pkl.dump(scalars, open(path_worker_dataset + '.pkl'), 'wb')

    def collect_trajs(self, nb_trajs, traj_collector, *params):
        def traj_collector_wrapper(seed):
            env = make_env(seed)
            params = [env] + params
            obss, actions = traj_collector(*params)
            self._store_traj(obss, actions, seed)

        print('Collecting trajectories {}-{}'.format(self.seed, self.seed + nb_trajs - 1))
        Parallel(n_jobs=self.nb_workers)(
            delayed(traj_collector_wrapper)(seed)
            for seed in tqdm(range(self.seed, self.seed + nb_trajs)))
        self.seed += nb_trajs

        print('Gathering trajectories dataset into one dataset...')
        gather_dataset(self.dataset_path)

    def collect_perfect_trajs(self, nb_trajs, expert, T):
        self.collect_trajs(nb_trajs, collect_perfect_traj, expert, T)

    def collect_corrected_trajs(self, nb_trajs, learner, expert, β, T):
        self.collect_trajs(nb_trajs, collect_corrected_traj, learner, expert, β, T)

def collect_perfect_traj(env, expert, T):
    obss = []
    actions = []

    obs = env.reset()
    expert.reset(env.dt, obs['cube_pos'], obs['goal_pos'])

    for _ in range(T):
        compress_images(obs)
        obss.append(obs)

        expert_action = expert.get_action(obs)
        actions.append(expert_action)

        obs, _, done, _ = env.step(expert_action)

        if done:
            break

    return obss, actions

def collect_corrected_traj(env, learner, expert, β, T):
    obss = []
    actions = []

    obs = env.reset()
    expert.reset(env.dt, obs['cube_pos'], obs['goal_pos'])

    for _ in range(T):
        compress_images(obs)
        obss.append(obs)

        expert_action = expert.get_action(obs)
        actions.append(expert_action)

        action = expert_action if th.rand() < β else learner.get_action(obs)
        obs, _, done, _ = env.step(action)

        if done:
            break

    return obss, actions