import pathlib

import gymnasium as gym
import numpy as np

# prepare environment
env_config = {
    "data_file": pathlib.Path(__file__).parent.resolve() / "data/data5.json",
    "stochastic": True,
    "render_mode": "human",
}
env = gym.make("mfgrl:mfgrl/MfgEnv-v0", env_config=env_config)
obs, info = env.reset()
print("Start state".center(100, "-"))
for k, v in env.decode_obs(obs).items():
    print(k, v)

# inference
total_reward = 0
steps = 0
max_production_action = np.argmax(env.decode_obs(obs)["market_production_rates"])
while True:
    steps += 1
    # action = max_production_action
    if steps < 10:
        action = np.random.randint(0, 5)
    else:
        action = 5
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    print("".center(100, "-"))
    print(
        f"i={steps}, a={action}, r={reward}, term={terminated}, info={info}, "
        f"tr={total_reward}"
    )
    for k, v in env.decode_obs(obs).items():
        print(k, v)
    if terminated:
        break

print(total_reward)
