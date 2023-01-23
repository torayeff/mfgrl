import pathlib

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
            "data_file": pathlib.Path(__file__).parent.resolve() / "data.json",
            "stochastic": True,
            "render_mode": None,
        },
    )
    .framework("torch")
    .training(grad_clip=0.5)
    .rollouts(num_rollout_workers=1)
    .resources(num_gpus=0)
    .evaluation(evaluation_num_workers=1, evaluation_interval=5)
)

# automated run
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={"timesteps_total": 500000},
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10),
    ),
)
results = tuner.fit()

# get best checkpoint
best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
print(f"Best checkpoint: {best_result.checkpoint}")
print(
    "Best episode reward mean: "
    f"{best_result.metrics['evaluation']['episode_reward_mean']}"
)

ray.shutdown()
