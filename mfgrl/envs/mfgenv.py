import gymnasium as gym
import numpy as np


class MfgEnv(gym.Env):
    def __init__(self, buffer_size=10):
        super().__init__()

        # max limit of configs
        self.buffer_size = buffer_size

        # hard limits
        self.MAX_TIME = int(10e9)
        self.MAX_PRODUCTS = int(10e9)

        # read this from json
        self.demand = 200
        self.demand_time = 100
        self.market_incurring_costs = np.array([1000.0, 1500.0, 2000.0, 500.0, 5000.0])
        self.market_recurring_costs = np.array([10.0, 15.0, 20.0, 5.0, 25.0])
        self.market_production_rates = np.array([1, 1.5, 2, 0.25, 5])
        self.market_setup_times = np.array([5, 7.5, 9, 3.5, 10])
        self.num_cfgs = len(self.market_incurring_costs)

        # observation and action spaces
        # TODO: encode/decode observation!!!
        self.observation_space = gym.spaces.Dict(
            {
                # demand data
                "demand": gym.spaces.Discrete(self.MAX_PRODUCTS),
                "demand_time": gym.spaces.Discrete(self.MAX_TIME),
                # available resources
                "incurred_costs": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.buffer_size,)
                ),
                "recurring_costs": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.buffer_size,)
                ),
                "production_rates": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.buffer_size,)
                ),
                "setup_times": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.buffer_size,)
                ),
                "cfgs_status": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.buffer_size,)
                ),
                "produced_counts": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.buffer_size,)
                ),
                # market data, changes dynamically
                "market_incurring_costs": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.num_cfgs,)
                ),
                "market_recurring_costs": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.num_cfgs,)
                ),
                "market_production_rates": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.num_cfgs,)
                ),
                "market_setup_times": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.num_cfgs,)
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(self.num_cfgs + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset buffer
        self.buffer_idx = 0

        # reset observation
        self.obs = {
            # demand data
            "demand": self.demand,
            "demand_time": self.demand_time,
            # available resources data
            "incurred_costs": np.zeros(self.buffer_size, dtype=np.float32),
            "recurring_costs": np.zeros(self.buffer_size, dtype=np.float32),
            "production_rates": np.zeros(self.buffer_size, dtype=np.float32),
            "setup_times": np.zeros(self.buffer_size, dtype=np.float32),
            "cfgs_status": np.zeros(self.buffer_size, dtype=np.float32),
            "produced_counts": np.zeros(self.buffer_size, dtype=np.float32),
            # market data
            "market_incurring_costs": self.market_incurring_costs,
            "market_recurring_costs": self.market_recurring_costs,
            "market_production_rates": self.market_production_rates,
            "market_setup_times": self.market_setup_times,
        }

        return self.obs, self._get_info()

    def step(self, action):
        if (0 <= action < self.num_cfgs) and (self.buffer_idx < self.buffer_size):
            reward = self.buy_cfg(cfg_id=action)
        else:
            reward = self.continue_production()

        terminated = self.obs["demand"] <= 0
        truncated = self.obs["demand_time"] <= 0

        if truncated:
            reward = -10e6

        return self.obs, reward, terminated, truncated, self._get_info()

    def buy_cfg(self, cfg_id):
        # calculate reward
        reward = -1.0 * self.obs["market_incurring_costs"][cfg_id]

        # buy new configuration; time is not updated
        self.obs["incurred_costs"][self.buffer_idx] = self.obs[
            "market_incurring_costs"
        ][cfg_id]

        self.obs["recurring_costs"][self.buffer_idx] = self.obs[
            "market_recurring_costs"
        ][cfg_id]

        self.obs["production_rates"][self.buffer_idx] = self.obs[
            "market_production_rates"
        ][cfg_id]

        self.obs["setup_times"][self.buffer_idx] = self.obs["market_setup_times"][
            cfg_id
        ]

        self.obs["cfgs_status"][self.buffer_idx] = (
            1 / self.obs["market_setup_times"][cfg_id]
        )

        self.obs["produced_counts"][self.buffer_idx] = 0

        self.buffer_idx += 1

        return reward

    def continue_production(self):
        # .astype(int) ensures that only ready machines contribute
        reward = -1.0 * np.sum(
            self.obs["cfgs_status"].astype(int) * self.obs["recurring_costs"]
        )

        # produce products with ready configurations
        self.obs["produced_counts"] += (
            self.obs["cfgs_status"].astype(int) * self.obs["production_rates"]
        )

        # update cfgs status
        # update only ready or being prepared cfgs
        updates = np.ceil(self.obs["cfgs_status"])
        # add small eps to deal with 0.999999xxx
        progress = [1 / st + 1e-9 if st != 0 else 0 for st in self.obs["setup_times"]]
        self.obs["cfgs_status"] = np.clip(
            self.obs["cfgs_status"] + updates * progress, a_min=0, a_max=1
        )

        # update observation
        self.obs["demand"] = self.demand - np.sum(
            self.obs["produced_counts"].astype(int)
        )
        self.obs["demand_time"] -= 1

        return reward

    def _get_info(self):
        return {}
