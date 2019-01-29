# Setup

- Install [mujoco](http://www.mujoco.org/)
- Install [mujoco-py](https://github.com/openai/mujoco-py)
- Clone [rstrudel/bcmuj](https://gitlab.inria.fr/rstrudel/bcmuj) and do `pip install -e .`
- Clone [rstrudel/bc](https://gitlab.inria.fr/rstrudel/bc) and do `pip install -e .`

# Commands

```text
python online_train.py [--method METHOD] [--resume RESUME]
```
- `METHOD` should be `bc`, `dagger` or `dart`
- `RESUME` can specify an epoch identifier to resume training from (saved models are stored in `./storage/models/[METHOD]/`)

```text
python eval.py [METHOD] [EPOCH] [--render] [--eps EPS] [--all ALL]
```
- `METHOD` should be `%expert`, `bc`, `dagger` or `dart`
- `EPOCH` should be the identifier of the epoch to evaluate (irrelevant for `%expert`)
- `--render` can be specified to render the environment
- `EPS` can specify a number of episodes to run (default is `1000`)
- `ALL` can specify to evaluate all epochs at the specified interval until `EPOCH` (not compatible with `--render`)

# Results

Our results can be found in [this notebook](Results.ipynb).

To reproduce, run:

```text
python online_train.py --method bc ;
python online_train.py --method dagger ;
python online_train.py --method dart ;
python eval.py bc 6144 --eps 500 --all 128 ;
python eval.py dagger 6144 --eps 500 --all 128 ;
python eval.py dart 6144 --eps 500 --all 128
```

Or just execute [the notebook](Results.ipynb).

*A GPU with a lot of memory is required to run this. It should take about 48 hours to train and evaluate.*