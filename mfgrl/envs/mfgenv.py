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
        self.setup_times = np.array([[5, 7.5, 9, 3.5, 10]]).T  # in time units

        self.observation_space = gym.spaces.Dict(
            {
                "cfgs_status": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(self.num_cfgs, self.cfgs_limit)
                ),
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
            "cfgs_status": np.zeros((self.num_cfgs, self.cfgs_limit), dtype=np.float32),
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
            # .astype(int) ensures that only ready machines contribute
            self.obs["produced"] += (
                self.obs["cfgs_status"].astype(int) * self.production_rates
            )
            reward = -1.0 * np.sum(
                self.obs["cfgs_status"].astype(int) * self.running_costs
            )

            # update cfgs status
            # add setup_times if 0 < cfg_status <= 1
            # add small eps to deal with 0.999999xxx
            # clip [0, 1]
            updates = np.ceil(self.obs["cfgs_status"])
            progress = 1 / self.setup_times + 1e-9
            self.obs["cfgs_status"] = np.clip(
                self.obs["cfgs_status"] + updates * progress, a_min=0, a_max=1
            )

            self._time_step += 1

        else:
            # buy new configuration and do not update time
            # find index with zero value
            idxs = np.where(self.obs["cfgs_status"][action] == 0)[0]
            if len(idxs) != 0:
                # initiate the installation of the configuration
                self.obs["cfgs_status"][action][idxs[0]] = (
                    1 / self.setup_times[action][0]
                )

            reward = -1.0 * self.purchase_costs[action][0]

        terminated = np.sum(self.obs["produced"].astype(int)) >= self.demand
        truncated = self._time_step >= self.duration

        return self.obs, reward, terminated, truncated, self._get_info()
