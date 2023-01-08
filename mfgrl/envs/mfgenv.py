import json
from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MfgEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_file: str,
        scale_costs: bool = True,
        stochastic: bool = False,
        render_mode: str = None,
    ):
        """Initialize

        Args:
            data_file (str): The data file location.
            scale_costs (book, optional): Whether to scale the costs. Defaults to True.
            stochastic (bool, optional): Stochastic environment. Defaults to False.
            render_mode (str, optional): Render mode. Defaults to None.
        """
        super().__init__()

        self.scale_costs = scale_costs
        self.stochastic = stochastic
        self.render_mode = render_mode

        if self.render_mode == "human":
            sns.set()
            plt.ion()

        self._setup_data(data_file)

        # observation and action spaces
        obs_dim = 2 + self.buffer_size * 6 + self.num_cfgs * 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,)
        )
        self.action_space = gym.spaces.Discrete(self.num_cfgs + 1)

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """Resets environment.

        Args:
            seed (int, optional): Random seed for determinism. Defaults to None.

        Returns:
            Tuple[np.ndarray, dict]: Observation and info.
        """
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
        # static state are used for stochastic operations
        self._static_state = {
            "recurring_costs": np.zeros(self.buffer_size, dtype=np.float32),
            "production_rates": np.zeros(self.buffer_size, dtype=np.float32),
        }

        if self.render_mode == "human":
            self._render_frame(action=-1, reward=0)
        else:
            print(self._env_state)

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Performs one step in environment.
        The environment time updates only if 0 <= action < self.num_cfgs

        If the environment is truncated high negative reward is returned.

        Args:
            action (int): Action index.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                Observation, reward, terminated, truncated, info.
        """
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
        else:
            print(self._env_state)

        if terminated or truncated:
            plt.show(block=True)
            plt.close("all")

        if self.stochastic:
            self._imitate_market_uncertainties()
            self._imitate_production_uncertainties()

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def buy_cfg(self, cfg_id: int) -> float:
        """Buys new configuration.
        This does not update the environment's time.

        Args:
            cfg_id (int): The index of new configuration.

        Returns:
            float: Reward as the negative cost of configuration.
        """
        # calculate reward
        reward = -1.0 * self._env_state["market_incurring_costs"][cfg_id]

        # buy new configuration
        # update incrred costs
        self._env_state["incurred_costs"][self.buffer_idx] = self._env_state[
            "market_incurring_costs"
        ][cfg_id]

        # update recurring costs
        self._env_state["recurring_costs"][self.buffer_idx] = self._env_state[
            "market_recurring_costs"
        ][cfg_id]
        self._static_state["recurring_costs"][self.buffer_idx] = self._env_state[
            "market_production_rates"
        ][cfg_id]

        # update production rates
        self._env_state["production_rates"][self.buffer_idx] = self._env_state[
            "market_production_rates"
        ][cfg_id]
        self._static_state["production_rates"][self.buffer_idx] = self._env_state[
            "market_production_rates"
        ][cfg_id]

        # update setup times
        self._env_state["setup_times"][self.buffer_idx] = self._env_state[
            "market_setup_times"
        ][cfg_id]

        # update cfgs status
        self._env_state["cfgs_status"][self.buffer_idx] = (
            1 / self._env_state["market_setup_times"][cfg_id]
        )

        # update production
        self._env_state["produced_counts"][self.buffer_idx] = 0

        # increment buffer idx
        self.buffer_idx += 1

        return reward * self.tradeoff

    def continue_production(self) -> float:
        """Continues production.
        This updates the environment's time.

        Returns:
            float: Reward as the sum of negative recurring costs.
        """
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

        return reward * (1 - self.tradeoff)

    def encode_obs(self, obs: dict) -> np.ndarray:
        """Encodes observation dictionary into vector.

        Args:
            obs (dict): Observation dictionary.

        Returns:
            np.ndarray: Observation vector.
        """
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

    def decode_obs(self, obs_vec: np.ndarray) -> dict:
        """Decodes observation vector into observation dictionary.

        Args:
            obs_vec (np.ndarray): Observation vector.

        Returns:
            dict: Observation dictionary.
        """
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

    def _imitate_production_uncertainties(self):
        """Imitates fluctuating production uncertainties:
        1. Failure of configurations: -+ 10%
        2. Production output: -+10%
        3. Recurring cost: -+10%
        """
        # imitate failure of a random configuration with failure rate 10%
        # failure changes cfg_status from 1 to random value between 0.7 and 0.1
        # where 0.7 is a major failure, and the value close to 1 is a minor failure
        # select randomly one of the running configurations
        running_cfgs = np.where(self._env_state["cfgs_status"] == 1)[0]
        if len(running_cfgs) > 0:
            cfg_id = np.random.choice(running_cfgs)
            if np.random.uniform(0, 1) > 0.9:
                self._env_state["cfgs_status"][cfg_id] = np.random.uniform(0.7, 1)

        # imitate fluctuating production rates
        prs = self._static_state["production_rates"]
        self._env_state["production_rates"][prs > 0] = np.random.uniform(
            low=(
                self._static_state["production_rates"][prs > 0]
                - 0.1 * self._static_state["production_rates"][prs > 0]
            ),
            high=(
                self._static_state["production_rates"][prs > 0]
                + 0.1 * self._static_state["production_rates"][prs > 0]
            ),
        )

        # imitate fluctuating recurring costs
        rcs = self._static_state["recurring_costs"]
        self._env_state["recurring_costs"][rcs > 0] = np.random.uniform(
            low=(
                self._static_state["recurring_costs"][rcs > 0]
                - 0.1 * self._static_state["recurring_costs"][rcs > 0]
            ),
            high=(
                self._static_state["recurring_costs"][rcs > 0]
                + 0.1 * self._static_state["recurring_costs"][rcs > 0]
            ),
        )

    def _imitate_market_uncertainties(self):
        """Imitates fluctuating market properties with 10% uncertainty."""
        # incurring costs
        self._env_state["market_incurring_costs"] = np.random.uniform(
            low=self.market_incurring_costs - 0.1 * self.market_incurring_costs,
            high=self.market_incurring_costs + 0.1 * self.market_incurring_costs,
        )

        # recurring costs
        self._env_state["market_recurring_costs"] = np.random.uniform(
            low=self.market_recurring_costs - 0.1 * self.market_recurring_costs,
            high=self.market_recurring_costs + 0.1 * self.market_recurring_costs,
        )

        # production rates
        self._env_state["market_production_rates"] = np.random.uniform(
            low=self.market_production_rates - 0.1 * self.market_production_rates,
            high=self.market_production_rates + 0.1 * self.market_production_rates,
        )

        # setup times
        self._env_state["market_setup_times"] = np.random.uniform(
            low=self.market_setup_times - 0.1 * self.market_setup_times,
            high=self.market_setup_times + 0.1 * self.market_setup_times,
        )

    def _get_obs(self) -> np.ndarray:
        """Gets observation.

        Returns:
            np.ndarray: Observation vector.
        """
        return self.encode_obs(self._env_state)

    def _get_info(self) -> dict:
        """Gets environment information.

        Returns:
            dict: Information.
        """
        return {}

    def _setup_data(self, data_file: str):
        """Sets up the data.

        Args:
            data_file (str): The location of data file.
        """
        with open(data_file, "r") as f:
            data = json.load(f)

        self.buffer_size = 10
        self.demand = data["demand"]
        self.demand_time = data["demand_time"]
        self.max_incurring_cost = data["max_incurring_cost"]
        self.max_recurring_cost = data["max_recurring_cost"]
        self.tradeoff = data["tradeoff"]
        self.num_cfgs = len(data["configurations"])

        self.market_incurring_costs = np.array([], dtype=np.float32)
        self.market_recurring_costs = np.array([], dtype=np.float32)
        self.market_production_rates = np.array([], dtype=np.float32)
        self.market_setup_times = np.array([], dtype=np.float32)

        for v in data["configurations"].values():
            self.market_incurring_costs = np.append(
                self.market_incurring_costs, v["incurring_cost"]
            )
            self.market_recurring_costs = np.append(
                self.market_recurring_costs, v["recurring_cost"]
            )
            self.market_production_rates = np.append(
                self.market_production_rates, v["production_rate"]
            )
            self.market_setup_times = np.append(
                self.market_setup_times, v["setup_time"]
            )

        if self.scale_costs:
            self.market_incurring_costs = (
                self.market_incurring_costs / self.max_incurring_cost
            )
            self.market_recurring_costs = (
                self.market_recurring_costs / self.max_recurring_cost
            )

    def _render_frame(self, **kwargs):
        """Renders one step of environment."""
        plt.close()

        data = self._env_state

        buffer_idxs = [f"B{i}" for i in range(self.buffer_size)]
        market_cfgs = [f"Mfg{i}" for i in range(self.num_cfgs)]
        fig, axes = plt.subplots(6, 2, figsize=(10, 7))
        palette = sns.color_palette()

        # remaining demand and time
        text_kwargs = dict(ha="center", va="center", fontsize=12)
        axes[0, 0].text(
            0.5,
            0.5,
            f"Remaining demand: {data['demand']}."
            f"Remaining time: {data['demand_time']}",
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
