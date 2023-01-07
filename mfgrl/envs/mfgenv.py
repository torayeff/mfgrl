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
        obs_dim = 2 + self.buffer_size * 6 + self.num_cfgs * 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,)
        )
        self.action_space = gym.spaces.Discrete(self.num_cfgs + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset buffer
        self.buffer_idx = 0

        # reset environment state
        self._env_state = {
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

        return self._get_obs(), self._get_info()

    def step(self, action):
        if (0 <= action < self.num_cfgs) and (self.buffer_idx < self.buffer_size):
            reward = self.buy_cfg(cfg_id=action)
        else:
            reward = self.continue_production()

        terminated = self._env_state["demand"] <= 0
        truncated = self._env_state["demand_time"] <= 0

        if truncated:
            reward = -10e6

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def buy_cfg(self, cfg_id):
        # calculate reward
        reward = -1.0 * self._env_state["market_incurring_costs"][cfg_id]

        # buy new configuration; time is not updated
        self._env_state["incurred_costs"][self.buffer_idx] = self._env_state[
            "market_incurring_costs"
        ][cfg_id]

        self._env_state["recurring_costs"][self.buffer_idx] = self._env_state[
            "market_recurring_costs"
        ][cfg_id]

        self._env_state["production_rates"][self.buffer_idx] = self._env_state[
            "market_production_rates"
        ][cfg_id]

        self._env_state["setup_times"][self.buffer_idx] = self._env_state[
            "market_setup_times"
        ][cfg_id]

        self._env_state["cfgs_status"][self.buffer_idx] = (
            1 / self._env_state["market_setup_times"][cfg_id]
        )

        self._env_state["produced_counts"][self.buffer_idx] = 0

        self.buffer_idx += 1

        return reward

    def continue_production(self):
        # .astype(int) ensures that only ready machines contribute
        reward = -1.0 * np.sum(
            self._env_state["cfgs_status"].astype(int)
            * self._env_state["recurring_costs"]
        )

        # produce products with ready configurations
        self._env_state["produced_counts"] += (
            self._env_state["cfgs_status"].astype(int)
            * self._env_state["production_rates"]
        )

        # update cfgs status
        # update only ready or being prepared cfgs
        updates = np.ceil(self._env_state["cfgs_status"])
        # add small eps to deal with 0.999999xxx
        progress = [
            1 / st + 1e-9 if st != 0 else 0 for st in self._env_state["setup_times"]
        ]
        self._env_state["cfgs_status"] = np.clip(
            self._env_state["cfgs_status"] + updates * progress, a_min=0, a_max=1
        )

        # update observation
        self._env_state["demand"] = self.demand - np.sum(
            self._env_state["produced_counts"].astype(int)
        )
        self._env_state["demand_time"] -= 1

        return reward

    def encode_obs(self, obs):
        return np.concatenate(
            (
                [obs["demand"], obs["demand_time"]],
                obs["incurred_costs"],
                obs["recurring_costs"],
                obs["production_rates"],
                obs["setup_times"],
                obs["cfgs_status"],
                obs["produced_counts"],
                obs["market_incurring_costs"],
                obs["market_recurring_costs"],
                obs["market_production_rates"],
                obs["market_setup_times"],
            )
        ).astype(np.float32)

    def decode_obs(self, obs_vec):
        obs_dict = {}
        obs_dict["demand"] = obs_vec[0]
        obs_dict["demand_time"] = obs_vec[1]

        start = 2
        obs_dict["incurred_costs"] = obs_vec[start : start + self.buffer_size]

        start += self.buffer_size
        obs_dict["recurring_costs"] = obs_vec[start : start + self.buffer_size]

        start += self.buffer_size
        obs_dict["production_rates"] = obs_vec[start : start + self.buffer_size]

        start += self.buffer_size
        obs_dict["setup_times"] = obs_vec[start : start + self.buffer_size]

        start += self.buffer_size
        obs_dict["cfgs_status"] = obs_vec[start : start + self.buffer_size]

        start += self.buffer_size
        obs_dict["produced_counts"] = obs_vec[start : start + self.buffer_size]

        start += self.buffer_size
        obs_dict["market_incurring_costs"] = obs_vec[start : start + self.num_cfgs]

        start += self.num_cfgs
        obs_dict["market_recurring_costs"] = obs_vec[start : start + self.num_cfgs]

        start += self.num_cfgs
        obs_dict["market_production_rates"] = obs_vec[start : start + self.num_cfgs]

        start += self.num_cfgs
        obs_dict["market_setup_times"] = obs_vec[start : start + self.num_cfgs]

        return obs_dict

    def _get_obs(self):
        return self.encode_obs(self._env_state)

    def _get_info(self):
        return {}
