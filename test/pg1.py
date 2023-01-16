from ray.rllib.algorithms.ppo import PPOConfig
from mfgrl.envs.mfgenv import MfgEnv

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
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [128, 128]})
    .evaluation(evaluation_num_workers=1)
)

algo = config.build()

for i in range(5):
    print(f"Training: {i}")
    algo.train()

print(algo.evaluate())
