import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MfgEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, buffer_size=10, render_mode=None):
        super().__init__()

        self.buffer_size = buffer_size
        self.render_mode = render_mode

        if self.render_mode == "human":
            sns.set()
            plt.ion()

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

        # total reward
        self.total_reward = 0

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

        if self.render_mode == "human":
            self._render_frame(action=-1, reward=0)

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

        self.total_reward += reward

        if self.render_mode == "human":
            self._render_frame(action=action, reward=reward)

        if terminated or truncated:
            plt.show(block=True)
            plt.close("all")

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

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

    def _render_frame(self, **kwargs):
        plt.close()

        data = self._env_state

        buffer_idxs = [f"B{i}" for i in range(self.buffer_size)]
        market_cfgs = [f"Mfg{i}" for i in range(self.num_cfgs)]
        fig, axes = plt.subplots(6, 2, figsize=(10, 7))
        palette = sns.color_palette()

        text_kwargs = dict(ha="center", va="center", fontsize=14, color=palette[3])
        # remaining demand and time
        axes[0, 0].text(
            0.5,
            0.5,
            f"Demand: {data['demand']}. Time: {data['demand_time']}",
            **text_kwargs,
        )
        axes[0, 0].set_yticklabels([])
        axes[0, 0].set_xticklabels([])
        axes[0, 0].grid(False)

        # cost
        action = kwargs["action"]
        reward = kwargs["reward"]
        axes[1, 0].text(
            0.5,
            0.5,
            f"Action: {action}. Cost: {reward}. Total cost: {-1.0 * self.total_reward}",
            **text_kwargs,
        )
        axes[1, 0].set_yticklabels([])
        axes[1, 0].set_xticklabels([])
        axes[1, 0].grid(False)

        # plot incurred costs
        axes[2, 0].bar(market_cfgs, data["market_incurring_costs"], color=palette[3])
        axes[2, 0].set_ylabel("$")
        axes[2, 0].set_xticklabels([])
        axes[2, 0].set_title("Incurring costs (market)")

        # plot recurring costs
        axes[3, 0].bar(market_cfgs, data["market_recurring_costs"], color=palette[4])
        axes[3, 0].set_ylabel("kWh")
        axes[3, 0].set_xticklabels([])
        axes[3, 0].set_title("Recurring costs (market)")

        # plot production rates
        axes[4, 0].bar(market_cfgs, data["market_production_rates"], color=palette[5])
        axes[4, 0].set_ylabel("p/h")
        axes[4, 0].set_xticklabels([])
        axes[4, 0].set_title("Production rates (market)")

        # plot setup times
        axes[5, 0].bar(market_cfgs, data["market_setup_times"], color=palette[6])
        axes[5, 0].set_ylabel("h")
        axes[5, 0].set_title("Setup times (market)")
        axes[5, 0].set_xlabel("Available configs.")

        # plot cfgs statuses
        progress_colors = [
            palette[2] if p == 1 else palette[1] for p in data["cfgs_status"]
        ]
        axes[0, 1].bar(buffer_idxs, data["cfgs_status"] * 100, color=progress_colors)
        axes[0, 1].set_ylabel("%")
        axes[0, 1].set_ylim([0, 100])
        axes[0, 1].set_xticklabels([])
        axes[0, 1].set_title("Configurations status (buffer)")

        # plot produced counts
        axes[1, 1].bar(buffer_idxs, data["produced_counts"], color=palette[0])
        axes[1, 1].set_ylabel("unit")
        axes[1, 1].set_ylim(bottom=0)
        axes[1, 1].set_xticklabels([])
        axes[1, 1].set_title("Production (buffer)")

        # plot incurred costs
        axes[2, 1].bar(buffer_idxs, data["incurred_costs"], color=palette[3])
        axes[2, 1].set_ylabel("h")
        axes[2, 1].set_xticklabels([])
        axes[2, 1].set_title("Incurred costs (buffer)")

        # plot recurring costs
        axes[3, 1].bar(buffer_idxs, data["recurring_costs"], color=palette[4])
        axes[3, 1].set_ylabel("kWh")
        axes[3, 1].set_xticklabels([])
        axes[3, 1].set_title("Recurring costs (buffer)")

        # plot production rates
        axes[4, 1].bar(buffer_idxs, data["production_rates"], color=palette[5])
        axes[4, 1].set_ylabel("p/h")
        axes[4, 1].set_xticklabels([])
        axes[4, 1].set_title("Production rates (buffer)")

        # plot setup times
        axes[5, 1].bar(buffer_idxs, data["setup_times"], color=palette[6])
        axes[5, 1].set_ylabel("h")
        axes[5, 1].set_title("Setup times (buffer)")
        axes[5, 1].set_xlabel("Available buffer")

        plt.tight_layout()

        fig.suptitle("Manufacturing Environment")
        fig.canvas.draw()
        fig.canvas.flush_events()
