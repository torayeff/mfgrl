import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MfgEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # read this from json
        self.demand = 200
        self.duration = 100

        num_cfgs = 5
        cfgs_limit = 10
        purchase_costs = np.array([1000.0, 1500.0, 2000.0, 500.0, 5000.0])  # in $
        running_costs = np.array([10.0, 15.0, 20.0, 5.0, 25.0])  # in $
        production_rates = np.array([1, 1.5, 2, 0.25, 5])  # per unit time

        # observation dictionary
        self.obs_dict = {
            "binary_map": np.zeros((num_cfgs, cfgs_limit)),
            "total_produced": np.zeros((num_cfgs, cfgs_limit)),
            "remaining_products": np.full((num_cfgs, cfgs_limit), self.demand),
            "remaining_time": np.full((num_cfgs, cfgs_limit), self.duration),
            "purchase_costs": np.repeat(
                purchase_costs.reshape(-1, 1), cfgs_limit, axis=1
            ),
            "running_costs": np.repeat(
                running_costs.reshape(-1, 1), cfgs_limit, axis=1
            ),
            "production_rates": np.repeat(
                production_rates.reshape(-1, 1), cfgs_limit, axis=1
            ),
        }

        # observation space is 7 by num_cfgs by cfgs_limit tensor
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(7, num_cfgs, cfgs_limit)
        )

        # action = self.num_cfgs: start/resume/continue production
        # 0 <= action < self.num_cfgs: buy configuration with index=action
        self.action_space = spaces.Discrete(num_cfgs + 1)

    def _get_obs(self):
        return self.obs_state

    def _get_info(self):
        return {"msg": "good luck!"}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_obs(), self._get_info()

    def step(self, action):
        if action == len(self.available_configs):
            # update remaining time
            self.obs_dict["remaining_time"] -= 1

            # update total produced
            self.obs_dict["total_produced"] += (
                self.obs_dict["binary_map"] * self.obs_dict["production_rates"]
            )

            # update remaining products
            self.obs_dict["remainig_products"] = self.demand - sum(
                self.obs_dict["total_produced"].astype(int)
            )
        else:
            # buy new configuration and do not update time

            # find non-zero index
            cfg = self.available_configs[action]
            idxs = np.where(cfg == 0)[0]
            if len(idxs) != 0:
                idx = idxs[0][0]
                self.available_configs[action, idx] = 1

                # reward = purchase cost
            else:
                # limit exceeded
                # give some negative reward
                pass

        obs = None
        reward = 0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    env = MfgEnv()
    obs, info = env.reset(seed=42)
    print(obs[5])
