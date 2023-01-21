import gymnasium as gym
import numpy as np

env_config = {
    "data_file": "/Users/torayeff/lab/mfgrl/test/data.json",
    "stochastic": True,
    "render_mode": "human",
}
env = gym.make("mfgrl:mfgrl/MfgEnv-v0", env_config=env_config)
obs, info = env.reset()
print("Start state".center(100, "-"))
for k, v in env.decode_obs(obs).items():
    print(k, v)

total_reward = 0
steps = 0
max_production_action = np.argmax(env.decode_obs(obs)["market_production_rates"])
while True:
    steps += 1
    action = max_production_action
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
