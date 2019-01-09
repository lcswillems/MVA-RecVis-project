import os

class Learner:
    def __init__(self, model_dir, trajs_dir):
        self.model_dir = model_dir
        self.trajs_dir = trajs_dir

    def train(self):
        os.system("python -m bc.net.train with net_config.json model.dir={} dataset.dir={}"
                  .format(self.model_dir, self.trajs_dir))