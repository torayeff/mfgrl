import pathlib

import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig

from mfgrl.envs.mfgenv import MfgEnv


# prepare config and load from checkpoint
# checkpoint_path = "./checkpoints/PPO_MfgEnv_checkpoint_000200"
# checkpoint_path = "./checkpoints/PPO_MfgEnv_checkpoint_001000"
# data 1
# checkpoint_path = "/Users/torayeff/ray_results/PPO/PPO_MfgEnv_2cd9b_00000_0_2023-01-24_11-28-07/checkpoint_000125"

# data 2
# checkpoint_path = "/Users/torayeff/ray_results/PPO/PPO_MfgEnv_a99a8_00000_0_2023-01-24_12-07-24/checkpoint_000250"

# data 3
# checkpoint_path = "/Users/torayeff/ray_results/PPO/PPO_MfgEnv_f7eab_00000_0_2023-01-24_13-49-48/checkpoint_000250"

# data 4
# checkpoint_path = "/Users/torayeff/ray_results/PPO/PPO_MfgEnv_eb2a2_00000_0_2023-01-24_14-39-33/checkpoint_000250"

# data 5
# checkpoint_path = "/Users/torayeff/ray_results/PPO/PPO_MfgEnv_691f3_00000_0_2023-01-24_15-11-43/checkpoint_000250"

config = (
    PPOConfig()
    .environment(
        MfgEnv,
        env_config={
            "data_file": pathlib.Path(__file__).parent.resolve() / "data/data.json",
            "stochastic": True,
            "render_mode": None,
        },
    )
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .evaluation(evaluation_num_workers=1)
    .resources(num_gpus=0)
)
algo = config.build()
algo.restore(checkpoint_path=checkpoint_path)

# prepare environment
env_config = {
    "data_file": pathlib.Path(__file__).parent.resolve() / "data/data.json",
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
