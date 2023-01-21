import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from mfgrl.envs.mfgenv import MfgEnv

# initialize ray
ray.init()

# config
config = (
    PPOConfig()
    .environment(
        MfgEnv,
        env_config={
            "data_file": "/Users/torayeff/lab/mfgrl/test/data.json",
            "stochastic": True,
            "render_mode": None,
        },
    )
    .framework("torch")
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
)

# automated run with Tune and grid search and TensorBoard
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={"training_iteration": 100},
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10),
    ),
)
results = tuner.fit()

# get best checkpoint
best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
print(f"Best checkpoint: {best_result.checkpoint}")

ray.shutdown()
