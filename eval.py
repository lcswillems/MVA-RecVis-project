import utils.env
import argparse
import time
from bc.agent.net_agent import NetAgent
import matplotlib.pyplot as plt

def main(args):
    env = utils.env.make_env(15623)
    agent = NetAgent(archi='sresnet10', channels='rgbd', path=args.net,
                     num_frames=1, skip=1, max_steps=300, epoch=args.epoch,
                     action_space='tool', dim_action=4, steps_action=1,
                     num_skills=1)

    plt.ion()
    fig = plt.figure()
    while True:
        obs = env.reset()
        done = False
        while not done:
            new_act = agent.get_action(dict(rgb=obs['rgb0'], depth=obs['depth0']))
            if new_act is not None:
                act = new_act

            print(act)
            obs, rew, done, _ = env.step(act)
            print(rew)
            fig.clear()
            plt.imshow(obs['rgb0'])
            plt.show()
            fig.canvas.flush_events()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            # time.sleep(.01)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('net', action='store', type=str)
    parser.add_argument('epoch', action='store', type=int)
    main(parser.parse_args())