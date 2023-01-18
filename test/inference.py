import gymnasium as gym
from ray.rllib.algorithms.algorithm import Algorithm

# load algorithm
checkpoint_path = "C:/Users/ezzat2/ray_results/PPO/PPO_MfgEnv_2e110_00000_0_2023-01-18_13-01-10/checkpoint_000100"
algo = Algorithm.from_checkpoint(checkpoint_path)

# prepare environment
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

# inference
total_reward = 0
steps = 0
while True:
    steps += 1

    action = algo.compute_single_action(obs)
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

print(f"Total reward: {total_reward}")
