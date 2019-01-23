import utils.env
import argparse
import utils
from utils.expert import PickPlaceExpert
import matplotlib.pyplot as plt
import matplotlib.image
import torch as th
import bc.net.zoo as zoo
import numpy as np
import tqdm

from concurrent.futures import ProcessPoolExecutor
import pickle

def compute(pars):
    return main(*pars)

def main(args, epoch, verbose=True):
    seed = 9218546
    env = utils.env.make_env(seed)
    if args.net != '%expert':
        net = zoo.Network(
            archi='resnet18', timesteps=3, input_type='rgbd',
            action_space='tool', dim_action=4,
            path='./storage/models/{}/resnet18_{}.pth'.format(args.net, epoch if epoch >= 0 else 'current'),
            steps_action=4, lam_grip=.1, device='cuda')

    expert = PickPlaceExpert()

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
        expert.reset(env.dt, obs['cube_pos'], obs['goal_pos'])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', action='store', type=str)
    parser.add_argument('epoch', action='store', type=int)
    parser.add_argument('--render', action='store_const', default=False, const=True)
    parser.add_argument('--save', action='store_const', default=False, const=True)
    parser.add_argument('--eps', action='store', default=1000, type=int)
    parser.add_argument('--all', action='store', default=0, type=int)
    args = parser.parse_args()
    if args.all > 0:
        epochs = np.arange(0, args.epoch, args.all) + args.all
        with ProcessPoolExecutor(max_workers=6) as executor:
            success = []
            for s in tqdm.tqdm(
                    executor.map(compute, [(args, epoch, False) for epoch in epochs]),
                    total=len(epochs)):
                success.append(s)
        print(success)
        pickle.dump(success, open('storage/success/{}'.format(args.net), 'wb'))
    else:
        main(args, args.epoch)