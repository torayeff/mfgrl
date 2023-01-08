# Manufacturing Reinforcement Learning Environment

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
import numpy as np
import random

np.random.seed(4242)
random.seed(4242)

env = gym.make(
    "mfgrl:mfgrl/MfgEnv-v0",
    data_file="data.json",
    stochastic=True,
    render_mode="human",
)
obs, info = env.reset(seed=42)

total_reward = 0
count = 0
while True:

    action = np.random.randint(0, env.action_space.n)

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

    count += 1

```
