import bc.net.zoo as zoo
import torch as th
import numpy as np
import torch.utils.data as data
import utils.env
import utils.expert
from concurrent.futures import ThreadPoolExecutor as Executor
import argparse
import utils
import tqdm
import os

seed = 0

steps_action = [0, 10, 20, 30]
timesteps = 3

epochs = 1024 * 3
episodes_per_epoch = 64

save_period = 64

p = .999

def main(args):
    save_dir = 'storage/models/{}'.format(args.method)
    os.makedirs(save_dir, exist_ok=True)

    net = zoo.Network(
        archi='resnet18', timesteps=timesteps, input_type='rgbd',
        action_space='tool', dim_action=4,
        steps_action=len(steps_action), lam_grip=.1,
        path='{}/resnet18_{}.pth'.format(save_dir, args.resume) if args.resume > 0 else None,
    )
    net.load_optimizer(learning_rate=1e-3, epochs=epochs)

    def optimize(obs, act, bound):
        loss_grip_list = []
        loss_move_list = []
        loader = data.DataLoader(
            utils.other.MultiFrameDataset(
                (obs, act),
                (-np.arange(timesteps)[::-1], steps_action),
                bound),
            batch_size=32
        )
        for obs, act in tqdm.tqdm(loader, total=len(loader), desc='optimize', position=1):
            net.optimizer.zero_grad()
            loss_grip, loss_move = net.compute_loss(dict(frames=obs), act)
            net.optimizer.step()
            loss_grip_list.append(loss_grip.item())
            loss_move_list.append(loss_move.item())
        return np.mean(loss_grip_list), np.mean(loss_move_list)

    with Executor(max_workers=1) as executor:
        env = executor.submit(utils.env.make_env, seed + args.resume * episodes_per_epoch).result()
        expert = utils.expert.PickPlaceExpert()

        if args.method == 'dagger':
            expert = utils.expert.DAggerExpert(expert, net)

        future = executor.submit(sample, env, expert, seed, episodes_per_epoch)
        loss_grip, loss_move = 0, 0
        progress = tqdm.trange(args.resume, epochs, desc='grip=?, move=?, success=?', position=0)
        for e in progress:
            epoch = e + 1
            if args.method == 'dagger':
                expert.β = p ** epoch
            trajectories, success = future.result()
            progress.set_description('grip={:.3f}, move={:.3f}, success={:.3f}'.format(loss_grip, loss_move, success))
            if epoch < epochs:
                future = executor.submit(sample, env, expert,
                                     seed + episodes_per_epoch * epoch, episodes_per_epoch)
            loss_grip, loss_move = optimize(*trajectories)

            if epoch % save_period == 0:
                net.save('{}/resnet18_{}.pth'.format(save_dir, epoch))
        net.save('{}/resnet18_{}.pth'.format(save_dir, 'current'))

def tensorize(trajectories):
    obs = []
    act = []
    trajectory_bound = []
    offset = 0
    for (obs_list, act_list) in tqdm.tqdm(trajectories, desc='tensorize', position=2):
        assert len(obs_list) == len(act_list)
        length = len(obs_list)
        trajectory_bound.append((offset, offset + length))
        obs.append(utils.other.transform_frames(obs_list))
        act.append(utils.other.transform_acts(act_list))
        offset += length

    bound = th.zeros((offset, 2), dtype=th.int).cuda()
    for start, end in trajectory_bound:
        bound[start:end, 0] = start
        bound[start:end, 1] = end

    return th.cat(obs), th.cat(act), bound

def sample(env, expert, seed, eps):
    mean_success = 0
    def episode(ep):
        obs_list = []
        act_list = []
        env.seed(seed + ep)
        obs = env.reset()
        expert.reset(env.dt, obs['cube_pos'], obs['goal_pos'])
        done = False
        frames = None
        while not done:
            frame = utils.other.transform_frames([obs])
            if frames is None:
                frames = frame.repeat(1, 3, 1, 1)
            else:
                frames[:, :-4] = frames[:, 4:]
                frames[:, -4:] = frame
            obs['frames'] = frames

            perfect_act, act = expert.act(obs)
            obs_list.append(obs)
            act_list.append(perfect_act)
            obs, _, done, success = env.step(act)

            if success['is_success']:
                return obs_list, act_list, True
        return obs_list, act_list, False

    trajectories = []
    for ep in tqdm.trange(eps, desc='sample', position=2):
        o, a, success = episode(ep)
        trajectories.append((o, a))
        mean_success += success / eps

    return tensorize(trajectories), mean_success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', action='store', default='bc', type=str)
    parser.add_argument('--resume', action='store', default=0, type=int)
    args = parser.parse_args()
    assert args.method in ['bc', 'dagger', 'dart']
    main(args)