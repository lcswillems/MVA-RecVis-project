import bc.net.zoo as zoo
import torch as th
import numpy as np
import torch.utils.data as data
from concurrent.futures import ThreadPoolExecutor as Executor
import argparse
import utils
import tqdm
import os
import copy

seed = 0

steps_action = [0, 10, 20, 30]
timesteps = 3

epochs = 1024 * 6
episodes_per_epoch = 32

save_period = 128

p = .9995
α = 3

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

    def optimize(obs, act, bound, compute_Σ=None):
        loss_grip_list = []
        loss_move_list = []
        dataset = utils.other.MultiFrameDataset(
                (obs, act),
                (-np.arange(timesteps)[::-1], steps_action),
                bound
        )
        loader = data.DataLoader(dataset, batch_size=32)
        for obs, act in tqdm.tqdm(loader, total=len(loader), desc='optimize', position=1):
            net.optimizer.zero_grad()
            loss_grip, loss_move = net.compute_loss(dict(frames=obs), act)
            net.optimizer.step()
            loss_grip_list.append(loss_grip.item())
            loss_move_list.append(loss_move.item())

        if compute_Σ is not None:
            del loader
            loader = data.DataLoader(dataset, batch_size=128)
            Σ = np.zeros((4, 4))
            for obs, act in tqdm.tqdm(loader, total=len(loader), desc='compute_Σ', position=1):
                pred = net(dict(frames=obs), compute_grad=False).cpu().numpy()
                pred_act = utils.other.pred_to_vector_act(pred)
                diff = pred_act - act[:, :4].cpu().numpy()
                Σ += (diff[:, :, None] * diff[:, None, :]).sum(axis=0)

            Σ = 1 / len(dataset)* Σ
            Σ = α * Σ / (len(dataset) * np.trace(Σ))
            compute_Σ(Σ)

        del dataset, loader, obs, act
        return np.mean(loss_grip_list), np.mean(loss_move_list)

    with Executor(max_workers=1) as executor:
        envs = [
            executor.submit(utils.env.make_env, seed + i + args.resume * episodes_per_epoch).result()
            for i in tqdm.trange(episodes_per_epoch, desc='init envs')
        ]
        expert = utils.expert.PickPlaceExpert(envs[0].dt)

        if args.method == 'dagger':
            expert = utils.expert.DAggerExpert(expert, copy.deepcopy(net))

        if args.method == 'dart':
            expert = utils.expert.GaussianExpert(expert)

        future = executor.submit(sample, envs, expert, seed, episodes_per_epoch)
        loss_grip, loss_move = 0, 0
        progress = tqdm.trange(args.resume, epochs, desc='grip=?, move=?, success=?', position=0)
        for e in progress:
            epoch = e + 1
            if args.method == 'dagger':
                expert.β = p ** epoch
                expert.net = copy.deepcopy(net)
            trajectories, success = future.result()
            progress.set_description('grip={:.3f}, move={:.3f}, success={:.3f}'.format(loss_grip, loss_move, success))
            if epoch < epochs:
                future = executor.submit(sample, envs, expert,
                                     seed + episodes_per_epoch * epoch, episodes_per_epoch)
            loss_grip, loss_move = optimize(*trajectories,
                                            compute_Σ=expert.set_Σ if args.method == 'dart' else None)

            if epoch % save_period == 0:
                net.save('{}/resnet18_{}.pth'.format(save_dir, epoch))
        net.save('{}/resnet18_{}.pth'.format(save_dir, 'current'))

def tensorize(trajectories):
    obs = []
    act = []
    trajectory_bound = []
    offset = 0
    for (frame_list, act_list) in trajectories:
        assert len(frame_list) == len(act_list)
        length = len(frame_list)
        trajectory_bound.append((offset, offset + length))
        obs.append(th.stack(frame_list))
        act.append(utils.other.transform_acts(act_list))
        offset += length

    bound = th.zeros((offset, 2), dtype=th.int).cuda()
    for start, end in trajectory_bound:
        bound[start:end, 0] = start
        bound[start:end, 1] = end

    return th.cat(obs), th.cat(act), bound

def sample(envs, expert, seed, eps):
    trajectories = [([], []) for _ in envs]
    mean_success = 0

    for i, env in enumerate(envs):
        env.seed(seed + i)
    obs = [env.reset() for env in envs]
    done = [False for _ in envs]
    frames = None
    for _ in tqdm.trange(50, desc='sample', position=2):
        frame = utils.other.transform_frames(obs)
        if frames is None:
            frames = frame.repeat(1, 3, 1, 1)
        else:
            frames[:, :-4] = frames[:, 4:]
            frames[:, -4:] = frame

        acts = expert.act_batch(obs, frames)
        for i, env in enumerate(envs):
            if done[i]:
                continue
            perfect_act, act = acts[i]
            trajectories[i][0].append(frame[i])
            trajectories[i][1].append(perfect_act)
            obs[i], _, done[i], s = env.step(act)
            if s['is_success']:
                done[i] = True
                mean_success += 1 / eps
        if all(done):
            break

    return tensorize(trajectories), mean_success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', action='store', default='bc', type=str)
    parser.add_argument('--resume', action='store', default=0, type=int)
    args = parser.parse_args()
    assert args.method in ['bc', 'dagger', 'dart']
    main(args)