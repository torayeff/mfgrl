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
import mfgrl

env = gym.make("mfgrl/MfgEnv-v0")

obs, info = env.reset(seed=42)
total_reward = 0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(total_reward)
```
