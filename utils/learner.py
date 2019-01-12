import os
import torch as th
from sacred import Experiment
from bc.net import log
from bc.net.train import train
from bc.net.utils import make_loader, make_net, write_info, get_train_loss
from bc.config import train_ingredient

class Learner:
    @train_ingredient.capture
    def __init__(self, model_dir, trajs_dir, ex_train, model, dataset):
        self.model_dir = model_dir
        self.trajs_dir = trajs_dir

        self.ex_train = ex_train
        self.ex_model = model
        self.ex_model['dir'] = model_dir
        self.ex_dataset = dataset
        self.ex_dataset['dir'] = self.trajs_dir

        self.net, self.ex_optimizer, self.ex_scheduler, self.ex_starting_epoch, self.ex_dir_net = make_net(model)

    def train(self):
        train_loader, eval_loader, statistics = make_loader(self.ex_model, self.ex_dataset)
        write_info(statistics)
        log.init_writers(self.ex_dir_net)
        train_loss = get_train_loss()
        train(train_loader, eval_loader, self.net, self.ex_dir_net, train_loss, self.ex_optimizer,
            self.ex_scheduler, self.ex_starting_epoch, self.ex_train)

    def act(self, obs):
        return self.net({'frames': obs})