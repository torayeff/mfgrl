import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig

from mfgrl.envs.mfgenv import MfgEnv

if __name__ == "__main__":
    ray.init(local_mode=True)

    config = (
        PPOConfig()
        .environment(
            MfgEnv,
            env_config={
                "data_file": "E:/lab/mfgrl/test/data.json",
                "stochastic": True,
                "render_mode": None,
            },
        )
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=1)
    )

    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop={"training_iteration": 1000},
            checkpoint_config=air.CheckpointConfig(checkpoint_frequency=10),
        ),
    )
    results = tuner.fit()

    # get best results
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    print(best_result.checkpoint)

    ray.shutdown()
