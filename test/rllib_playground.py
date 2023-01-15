import random

import numpy as np
from mfgenv_gym import MfgEnv
from ray.rllib.algorithms.ppo import PPOConfig

np.random.seed(4242)
random.seed(4242)


env_config = {
    "data_file": "data.json",
    "scale_costs": True,
    "stochastic": False,
    "render_mode": None,
}


config = (
    PPOConfig()
    .environment(env=MfgEnv, env_config=env_config)
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
    .evaluation(evaluation_num_workers=1)
)

algo = config.build()

for i in range(5):
    print(i)
    algo.train()

print("Evaluation".center(100, "-"))
print(algo.evaluate())
