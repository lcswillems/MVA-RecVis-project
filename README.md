# Setup

- Install [mujoco](http://www.mujoco.org/)
- Install [mujoco-py](https://github.com/openai/mujoco-py)
- Clone [rstrudel/bcmuj](https://gitlab.inria.fr/rstrudel/bcmuj) and do `pip install -e .`
- Clone [rstrudel/bc](https://gitlab.inria.fr/rstrudel/bc) and do `pip install -e .`

# Commands to try

## 1. Generate 20 demos

```
python -m bcmuj.dataset.collect_data --agent script --dataset storage/demos --episodes 20 --processes 1
```

## 2.Train a model by BC

```
python -m bc.net.train with config_bc.json
```

## 3. Test the BC-trained model

```
python -m bcmuj.dataset.collect_data --agent net --net_path storage/models/bc --seed 50000 --episodes 100
--first_epoch 2 --last_epoch 12 --iter_epoch 4 --report_path storage/models/bc/test --video_path storage/models/bc/test --processes 8
```