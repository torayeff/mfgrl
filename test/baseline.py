import gymnasium as gym
import numpy as np
import random

np.random.seed(42)
random.seed(42)

env = gym.make("mfgrl:mfgrl/MfgEnv-v0", render_mode="human")
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
