import json

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MfgEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_file, stochastic_market=False, render_mode=None):
        super().__init__()

        self.stochastic_market = stochastic_market
        self.render_mode = render_mode

        if self.render_mode == "human":
            sns.set()
            plt.ion()

        self._setup_data(data_file)

        # observation and action spaces
        obs_dim = 2 + self.buffer_size * 7 + self.num_cfgs * 5
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
            "up_times": np.zeros(self.buffer_size, dtype=np.float32),
            "cfgs_status": np.zeros(self.buffer_size, dtype=np.float32),
            "produced_counts": np.zeros(self.buffer_size, dtype=np.float32),
            # market data
            "market_incurring_costs": self.market_incurring_costs,
            "market_recurring_costs": self.market_recurring_costs,
            "market_production_rates": self.market_production_rates,
            "market_setup_times": self.market_setup_times,
            "market_up_times": self.market_up_times,
        }

        if self.render_mode == "human":
            self._render_frame(action=-1, reward=0)
        else:
            print(self._env_state)

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

        if self.stochastic_market:
            self._update_market()

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

        self._env_state["up_times"][self.buffer_idx] = self._env_state[
            "market_up_times"
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

    def _update_market(self):
        # change -+10%
        # clip between -+20% of initial costs

        # incurring costs
        self._env_state["market_incurring_costs"] += np.random.uniform(
            low=-0.1 * self._env_state["market_incurring_costs"],
            high=0.1 * self._env_state["market_incurring_costs"],
        )
        self._env_state["market_incurring_costs"] = np.clip(
            self._env_state["market_incurring_costs"],
            a_min=self.market_incurring_costs - 0.2 * self.market_incurring_costs,
            a_max=self.market_incurring_costs + 0.2 * self.market_incurring_costs,
        )

        # recurring costs
        self._env_state["market_recurring_costs"] += np.random.uniform(
            low=-0.1 * self._env_state["market_recurring_costs"],
            high=0.1 * self._env_state["market_recurring_costs"],
        )
        self._env_state["market_recurring_costs"] = np.clip(
            self._env_state["market_recurring_costs"],
            a_min=self.market_recurring_costs - 0.2 * self.market_recurring_costs,
            a_max=self.market_recurring_costs + 0.2 * self.market_recurring_costs,
        )

        # production rates
        self._env_state["market_production_rates"] += np.random.uniform(
            low=-0.1 * self._env_state["market_production_rates"],
            high=0.1 * self._env_state["market_production_rates"],
        )
        self._env_state["market_production_rates"] = np.clip(
            self._env_state["market_production_rates"],
            a_min=self.market_production_rates - 0.2 * self.market_production_rates,
            a_max=self.market_production_rates + 0.2 * self.market_production_rates,
        )

        # setup times
        self._env_state["market_setup_times"] += np.random.uniform(
            low=-0.1 * self._env_state["market_setup_times"],
            high=0.1 * self._env_state["market_setup_times"],
        )
        self._env_state["market_setup_times"] = np.clip(
            self._env_state["market_setup_times"],
            a_min=self.market_setup_times - 0.2 * self.market_setup_times,
            a_max=self.market_setup_times + 0.2 * self.market_setup_times,
        )

        # setup up_times
        self._env_state["market_up_times"] += np.random.uniform(
            low=-0.1 * self._env_state["market_up_times"],
            high=0.1 * self._env_state["market_up_times"],
        )
        self._env_state["market_up_times"] = np.clip(
            self._env_state["market_up_times"],
            a_min=self.market_up_times - 0.2 * self.market_up_times,
            a_max=self.market_up_times + 0.2 * self.market_up_times,
        )

    def encode_obs(self, obs):
        return np.concatenate(
            (
                [obs["demand"], obs["demand_time"]],
                obs["incurred_costs"],
                obs["recurring_costs"],
                obs["production_rates"],
                obs["setup_times"],
                obs["up_times"],
                obs["cfgs_status"],
                obs["produced_counts"],
                obs["market_incurring_costs"],
                obs["market_recurring_costs"],
                obs["market_production_rates"],
                obs["market_setup_times"],
                obs["market_up_times"],
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
        obs_dict["up_times"] = obs_vec[start : start + self.buffer_size]

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

        start += self.num_cfgs
        obs_dict["market_up_times"] = obs_vec[start : start + self.num_cfgs]

        return obs_dict

    def _get_obs(self):
        return self.encode_obs(self._env_state)

    def _get_info(self):
        return {}

    def _setup_data(self, data_file):
        with open(data_file, "r") as f:
            data = json.load(f)

        self.buffer_size = 10
        self.demand = data["demand"]
        self.demand_time = data["demand_time"]
        self.num_cfgs = len(data["configurations"])

        self.market_incurring_costs = []
        self.market_recurring_costs = []
        self.market_production_rates = []
        self.market_setup_times = []
        self.market_up_times = []

        for k, v in data["configurations"].items():
            self.market_incurring_costs.append(v["incurring_cost"])
            self.market_recurring_costs.append(v["recurring_cost"])
            self.market_production_rates.append(v["production_rate"])
            self.market_setup_times.append(v["setup_time"])
            self.market_up_times.append(v["up_time"])

        self.market_incurring_costs = np.array(
            self.market_incurring_costs, dtype=np.float32
        )
        self.market_recurring_costs = np.array(
            self.market_recurring_costs, dtype=np.float32
        )
        self.market_production_rates = np.array(
            self.market_production_rates, dtype=np.float32
        )
        self.market_setup_times = np.array(self.market_setup_times, dtype=np.float32)
        self.market_up_times = np.array(self.market_up_times, dtype=np.float32)

    def _render_frame(self, **kwargs):
        plt.close()

        data = self._env_state

        buffer_idxs = [f"B{i}" for i in range(self.buffer_size)]
        market_cfgs = [f"Mfg{i}" for i in range(self.num_cfgs)]
        fig, axes = plt.subplots(6, 2, figsize=(10, 7))
        palette = sns.color_palette()

        # remaining demand and time
        text_kwargs = dict(ha="center", va="center", fontsize=14)
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
        text_kwargs = dict(ha="center", va="center", fontsize=12)
        action = kwargs["action"]
        reward = kwargs["reward"]
        axes[1, 0].text(
            0.5,
            0.5,
            f"Action: {action}. Step cost: {-reward:.1f}."
            f" Total cost: {-1.0 * self.total_reward:.1f}",
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
        axes[1, 1].set_ylim([0, self.demand + 1])
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
