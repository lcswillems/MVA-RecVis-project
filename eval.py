import argparse
import utils
import matplotlib.pyplot as plt
import matplotlib.image
import bc.net.zoo as zoo
import numpy as np
import tqdm

import pickle

def load_net(args, epoch):
    return zoo.Network(
        archi='resnet18', timesteps=3, input_type='rgbd',
        action_space='tool', dim_action=4,
        path='./storage/models/{}/resnet18_{}.pth'.format(args.net, epoch if epoch >= 0 else 'current'),
        steps_action=4, lam_grip=.1, device='cuda')

def main(args, epoch, verbose=True):
    if args.net != '%expert':
        net = load_net(args, epoch)

    return sequential(args, args.seed, net, verbose)

def sequential(args, seed, net, verbose):
    env = utils.env.make_env(seed)
    expert = utils.expert.PickPlaceExpert(env.dt)

    if args.render:
        plt.ion()
        fig = plt.figure()

    mean_ep_rew = 0
    mean_ep_err = 0
    mean_success = 0
    eps = args.eps
    frame = 0
    for ep in range(eps):
        ep_rew = 0
        ep_err = 0
        env.seed(seed + ep)
        obs = env.reset()
        done = False
        if args.net != '%expert':
            frames = None
        while not done:
            if args.net != '%expert':
                frame = utils.other.transform_frames([obs])
                if frames is None:
                    frames = frame.repeat(1, 3, 1, 1)
                else:
                    frames[:, :-4] = frames[:, 4:]
                    frames[:, -4:] = frame
                obs['frames'] = frames
                act = net.get_dic_action(obs)
                ep_err += ((utils.other.va(act) - utils.other.va(expert.act(obs)[0])) ** 2).sum()
            else:
                _, act = expert.act(obs)
            obs, rew, done, success = env.step(act)

            ep_rew += rew

            if success['is_success']:
                mean_success += 1 / eps
                done = True

            if args.save:
                matplotlib.image.imsave('storage/render/{}.png'.format(frame), obs['rgb0'])
                frame += 1
            if args.render:
                fig.clear()
                plt.imshow(obs['rgb0'])
                plt.axis('off')
                if args.render:
                    plt.show()
                    fig.canvas.flush_events()
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
        if verbose:
            print('ep {} done | rews={}, errs={}, success={}'.format(ep, ep_rew, ep_err, success['is_success']))
        mean_ep_rew += ep_rew / eps
        mean_ep_err += ep_err / eps
    if verbose:
        print('mean | rews={}, errs={}, succs={}'.format(mean_ep_rew, mean_ep_err, mean_success))
    return mean_success

def concurrent(args, epoch, envs):
    expert = utils.expert.DAggerExpert(utils.expert.PickPlaceExpert(envs[0].dt), load_net(args, epoch))
    expert.β = 0
    mean_success = 0

    for i, env in enumerate(envs):
        env.seed(args.seed + i)
    obs = [env.reset() for env in envs]
    done = [False for _ in envs]
    frames = None
    for _ in tqdm.trange(50, desc='sample', position=0):
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
            obs[i], _, done[i], s = env.step(act)
            if s['is_success']:
                done[i] = True
                mean_success += 1 / args.eps
        if all(done):
            break
    return mean_success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', action='store', type=str)
    parser.add_argument('epoch', action='store', type=int)
    parser.add_argument('--render', action='store_const', default=False, const=True)
    parser.add_argument('--save', action='store_const', default=False, const=True)
    parser.add_argument('--eps', action='store', default=1000, type=int)
    parser.add_argument('--all', action='store', default=0, type=int)
    parser.add_argument('--seed', action='store', default=9218546, type=int)
    args = parser.parse_args()
    if args.all > 0:
        epochs = np.arange(0, args.epoch, args.all) + args.all
        success = []
        envs = [
            utils.env.make_env(args.seed + i)
            for i in tqdm.trange(args.eps, desc='init envs')
        ]
        for epoch in tqdm.tqdm(epochs, total=len(epochs), position=1):
            success.append(concurrent(args, epoch, envs))
        print(success)
        pickle.dump(success, open('storage/success/{}'.format(args.net), 'wb'))
    else:
        main(args, args.epoch)