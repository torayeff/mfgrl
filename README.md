# Manufacturing Reinforcement Learning Environment
![](docs/mfgrl_vis.gif)

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
