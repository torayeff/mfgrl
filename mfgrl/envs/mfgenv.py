import gymnasium as gym
import numpy as np


class MfgEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # hard limits
        self.MAX_TIME = int(10e9)
        self.MAX_PRODUCTS = int(10e9)
        self.MAX_BUFFER_SIZE = 10

        # read this from json
        self.demand = 200
        self.duration = 100
        self.purchase_costs = np.array([1000.0, 1500.0, 2000.0, 500.0, 5000.0])
        self.running_costs = np.array([10.0, 15.0, 20.0, 5.0, 25.0])
        self.production_rates = np.array([1, 1.5, 2, 0.25, 5])
        self.setup_times = np.array([5, 7.5, 9, 3.5, 10])
        self.num_cfgs = len(self.purchase_costs)

        # observation and action spaces
        self.observation_space = gym.spaces.Dict(
            {
                "duration": gym.spaces.Discrete(self.MAX_TIME),
                "demand": gym.spaces.Discrete(self.MAX_PRODUCTS),
                "purchase_costs": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.num_cfgs,)
                ),
                "running_costs": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.num_cfgs,)
                ),
                "production_rates": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.num_cfgs,)
                ),
                "setup_times": gym.spaces.Box(
                    low=0.0, high=10e9, shape=(self.num_cfgs,)
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(self.num_cfgs + 1)

    def __reset_all(self):
        # reset time
        self.current_time = 0

        # reset buffer
        self.buffer = {
            "purchase_costs": np.array([], dtype=np.float32),
            "running_costs": np.array([], dtype=np.float32),
            "production_rates": np.array([], dtype=np.float32),
            "setup_times": np.array([], dtype=np.float32),
            "cfgs_status": np.array([], dtype=np.float32),
            "produced": np.array([], dtype=np.float32),
        }

        # reset observation
        self.obs = {
            "duration": self.duration,
            "demand": self.demand,
            "purchase_costs": self.purchase_costs,
            "running_costs": self.running_costs,
            "production_rates": self.production_rates,
            "setup_times": self.setup_times,
        }

    def _get_info(self):
        return {"msg": "good luck!"}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.__reset_all()
        return self.obs, self._get_info()

    def step(self, action):
        if action < self.num_cfgs:
            # buy new configuration; time is not updated
            reward = -1.0 * self.purchase_costs[action]
            self.buffer["purchase_costs"] = np.append(
                self.buffer["purchase_costs"], self.obs["purchase_costs"][action]
            )
            self.buffer["running_costs"] = np.append(
                self.buffer["running_costs"], self.obs["running_costs"][action]
            )
            self.buffer["production_rates"] = np.append(
                self.buffer["production_rates"], self.obs["production_rates"][action]
            )
            self.buffer["setup_times"] = np.append(
                self.buffer["setup_times"], self.obs["setup_times"][action]
            )
            self.buffer["cfgs_status"] = np.append(
                self.buffer["cfgs_status"], 1 / self.obs["setup_times"][action]
            )
            self.buffer["produced"] = np.append(self.buffer["produced"], 0)
        else:
            # .astype(int) ensures that only ready machines contribute
            reward = -1.0 * np.sum(
                self.buffer["cfgs_status"].astype(int) * self.buffer["running_costs"]
            )

            # produce products with ready configurations
            self.buffer["produced"] += (
                self.buffer["cfgs_status"].astype(int) * self.buffer["production_rates"]
            )

            # update cfgs status
            # update only ready or being prepared cfgs
            updates = np.ceil(self.buffer["cfgs_status"])
            # add small eps to deal with 0.999999xxx
            progress = 1 / self.buffer["setup_times"] + 1e-9
            self.buffer["cfgs_status"] = np.clip(
                self.buffer["cfgs_status"] + updates * progress, a_min=0, a_max=1
            )

            # update observation
            self.obs["demand"] = self.demand - np.sum(
                self.buffer["produced"].astype(int)
            )
            self.obs["duration"] -= 1

        terminated = self.obs["demand"] <= 0
        truncated = self.obs["duration"] <= 0

        return self.obs, reward, terminated, truncated, self._get_info()
