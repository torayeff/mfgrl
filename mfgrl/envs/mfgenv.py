import gymnasium as gym
import numpy as np


class MfgEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # read this from json
        self.demand = 200
        self.duration = 100
        self.num_cfgs = 5
        self.cfgs_limit = 10
        self.purchase_costs = np.array(
            [[1000.0, 1500.0, 2000.0, 500.0, 5000.0]]
        ).T  # in $
        self.running_costs = np.array([[10.0, 15.0, 20.0, 5.0, 25.0]]).T  # in $
        self.production_rates = np.array([[1, 1.5, 2, 0.25, 5]]).T  # per unit time

        self.observation_space = gym.spaces.Dict(
            {
                "cfgs_mask": gym.spaces.MultiBinary((self.num_cfgs, self.cfgs_limit)),
                "produced": gym.spaces.Box(
                    low=0.0, high=10e6, shape=(self.num_cfgs, self.cfgs_limit)
                ),
            }
        )

        # action = self.num_cfgs: start/resume/continue production
        # 0 <= action < self.num_cfgs: buy configuration with index=action
        self.action_space = gym.spaces.Discrete(self.num_cfgs + 1)

    def _reset_initial_obs(self):
        self.obs = {
            "cfgs_mask": np.zeros((self.num_cfgs, self.cfgs_limit), dtype=np.float32),
            "produced": np.zeros((self.num_cfgs, self.cfgs_limit), dtype=np.float32),
        }

    def _get_info(self):
        return {"msg": "good luck!"}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_initial_obs()
        self._time_step = 0

        return self.obs, self._get_info()

    def step(self, action):
        if action == self.num_cfgs:
            # produce products
            self.obs["produced"] += self.obs["cfgs_mask"] * self.production_rates
            self._time_step += 1
            reward = -1.0 * np.sum(self.obs["cfgs_mask"] * self.running_costs)
        else:
            # buy new configuration and do not update time
            # find index with zero value
            idxs = np.where(self.obs["cfgs_mask"][action] == 0)[0]
            if len(idxs) != 0:
                self.obs["cfgs_mask"][action][idxs[0]] = 1

            reward = -1.0 * self.purchase_costs[action][0]

        terminated = np.sum(self.obs["produced"].astype(int)) >= self.demand
        truncated = self._time_step >= self.duration

        return self.obs, reward, terminated, truncated, self._get_info()
