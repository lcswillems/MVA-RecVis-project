# Setup

- Install [mujoco](http://www.mujoco.org/)
- Install [mujoco-py](https://github.com/openai/mujoco-py)
- Clone [rstrudel/bcmuj](https://gitlab.inria.fr/rstrudel/bcmuj) and do `pip install -e .`
- Clone [rstrudel/bc](https://gitlab.inria.fr/rstrudel/bc) and do `pip install -e .`

# Commands to try

```
python train.py with net_config.json algo=bc
```