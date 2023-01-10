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
max_production_action = np.argmax(env.decode_obs(obs)["market_production_rates"])
while True:
    count += 1

    if len(np.where(env.decode_obs(obs)["incurred_costs"] == 0)[0]) == 0:
        # if the buffer is full continue production
        action = env.action_space.n
    else:
        # buy randomly until the buffer is full
        # action = np.random.randint(0, env.action_space.n - 1)
        action = max_production_action
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"i={count}, a={action}, r={reward}, term={terminated}, info={info}")
    print(env.decode_obs(obs))
    total_reward += reward
    if terminated:
        break

print(total_reward)
