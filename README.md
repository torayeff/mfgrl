# Manufacturing Reinforcement Learning Environment
## Acknowledgements
This work was developed at the [Institute for Advanced Manufacturing at the University of Nottingham](https://www.nottingham.ac.uk/ifam/index.aspx), in collaboration with the [Software and systems engineering](https://www.mondragon.edu/en/research-transfer/engineering-technology/research-and-transfer-groups/-/mu-inv-mapping/grupo/ingenieria-del-sw-y-sistemas) and the [High-performance machining](https://www.mondragon.edu/en/research-transfer/engineering-technology/research-and-transfer-groups/-/mu-inv-mapping/grupo/mecanizado-de-alto-rendimiento) groups at [Mondragon University](https://www.mondragon.edu/en/home), as part of the Digital Manufacturing and Design Training Network.

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 814078.

![](docs/mfgrl_vis.gif)
Please refer to this [file](docs/description.md) for problem formulation and the environment description.

## Installation
Clone
```
git clone git@github.com:torayeff/mfgrl.git
```

Install
```
pip install -e mfgrl
```

## Usage
```python
import gymnasium as gym

env_config = {
    "data_file": "data.json",
    "stochastic": True,
    "render_mode": "human",
}
env = gym.make("mfgrl:mfgrl/MfgEnv-v0", env_config=env_config)
obs, info = env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```
