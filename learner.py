import torch.nn as nn
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader

def get_learner_generator(lr, grip_coef, batch_size, epochs, eval_prop, eval_interval):
    def gen_learner():
        return Learner(lr, grip_coef, batch_size, epochs, eval_prop, eval_interval)
    return gen_learner

class Learner:
    def __init__(self, lr, grip_coef, batch_size, epochs, eval_prop, eval_interval):
        self.lr = lr
        self.grip_coef = grip_coef
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_prop = eval_prop
        self.eval_interval = eval_interval

        self.policy = models.resnet18()
        self.l2_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def train(self, trajs):
        obss, acts = trajs
        dataset = TensorDataset(obss, acts)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for obss, acts in dataloader:
                preds = self.policy(obss)
                l2_loss = self.l2_loss()

        #TODO:
        raise NotImplementedError

    def evaluate(self, trajs):

        #TODO:
        raise NotImplementedError

    def act(self, obs):

        #TODO:
        raise NotImplementedError