import gymnasium as gym
import numpy as np
import random

np.random.seed(42421)
random.seed(42421)

env = gym.make(
    "mfgrl:mfgrl/MfgEnv-v0",
    data_file="data.json",
    stochastic=True,
    render_mode="human",
)
obs, info = env.reset(seed=421)

total_reward = 0
count = 0
while True:
    count += 1

    action = np.random.randint(0, env.action_space.n)
    action = 0

    obs, reward, terminated, truncated, info = env.step(action)
    print("Step: ", count, "action: ", action, reward, terminated, truncated)
    print(env.decode_obs(obs))
    total_reward += reward
    if terminated or truncated:
        break
