import utils.env
import argparse
from utils import va
from utils.expert import PickPlaceExpert
from bc.agent.net_agent import NetAgent
import matplotlib.pyplot as plt
import matplotlib.image

def main(args):
    seed = 1792 + 998
    env = utils.env.make_env(seed)
    agent = NetAgent(archi='sresnet10', channels='rgbd', path=args.net,
                     num_frames=1, skip=1, max_steps=-1, epoch=args.epoch,
                     action_space='tool', dim_action=4, steps_action=1,
                     num_skills=1)

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
        while not done:
            obs_dict = dict(rgb=obs['rgb0'], depth=obs['depth0'])
            act = agent.get_action(obs_dict)
            obs, rew, done, success = env.step(act)

            ep_rew += rew
            ep_err += ((va(act) - va(expert.act(obs)[0])) ** 2).sum()
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
        print('ep {} done | rews={}, errs={}, success={}'.format(ep, ep_rew, ep_err, success['is_success']))
        mean_ep_rew += ep_rew / eps
        mean_ep_err += ep_err / eps
    print('mean | rews={}, errs={}, succs={}'.format(mean_ep_rew, mean_ep_err, mean_success))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', action='store', type=str)
    parser.add_argument('epoch', action='store', type=int)
    parser.add_argument('--render', action='store_const', default=False, const=True)
    parser.add_argument('--save', action='store_const', default=False, const=True)
    parser.add_argument('--eps', action='store', default=1000, type=int)
    main(parser.parse_args())