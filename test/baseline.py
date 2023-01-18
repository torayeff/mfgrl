import gymnasium as gym
import numpy as np
import random

np.random.seed(4242)
random.seed(4242)

env_config = {
    "data_file": "data.json",
    "scale_costs": True,
    "stochastic": False,
    "render_mode": "human",
}
env = gym.make("mfgrl:mfgrl/MfgEnv-v0", env_config=env_config)
obs, info = env.reset(seed=42)
print("Start state".center(100, "-"))
for k, v in env.decode_obs(obs).items():
    print(k, v)

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
        action = max_production_action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    print("".center(100, "-"))
    print(
        f"i={count}, a={action}, r={reward}, term={terminated}, info={info}, "
        f"tr={total_reward}"
    )
    for k, v in env.decode_obs(obs).items():
        print(k, v)
    if terminated:
        break

print(total_reward)
