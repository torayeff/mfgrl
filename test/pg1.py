from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

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

for i in range(100):
    print(f"Training: {i}")
    result = algo.train()
    print(pretty_print(result))

# save
checkpoint_dir = algo.save()
print(f"Checkpoint saved in directory {checkpoint_dir}")

print(algo.evaluate())
