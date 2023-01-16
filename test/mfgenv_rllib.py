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
                "scale_costs": True,
                "stochastic": False,
                "render_mode": None,
            },
        )
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        .resources(num_gpus=1)
    )

    stop = {
        "training_iteration": 500,
        "timesteps_total": 10e6,
        "episode_reward_mean": 90,
    }

    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop),
    )
    results = tuner.fit()

    ray.shutdown()
