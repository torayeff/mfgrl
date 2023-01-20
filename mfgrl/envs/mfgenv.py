import json
from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class MfgEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config: dict):
        """Initialize the environment.

        Args:
            env_config (dict): The configuration dictionary.
                    It must have the following keys and values:
                    data_file (str): The data file location.
                    stochastic (bool, optional): Stochastic environment.
                                                 Defaults to False.
                    render_mode (str, optional): Render mode. Defaults to None.
        """
        super().__init__()

        self.BUFFER_SIZE = env_config["buffer_size"]
        self.NUM_CFGS = env_config["num_cfgs"]
        self.DATA_FILE = env_config["data_file"]
        self.STOCHASTIC = env_config["stochastic"]
        self.RENDER_MODE = env_config["render_mode"]

        obs_dim = 2 + self.BUFFER_SIZE * 6 + self.NUM_CFGS * 4
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,)
        )
        self.action_space = gym.spaces.Discrete(self.NUM_CFGS + 1)

    def reset(self, seed: int = None, options: dict = None) -> Tuple[np.ndarray, dict]:
        """Resets environment.

        Args:
            seed (int, optional): Random seed for determinism. Defaults to None.

        Returns:
            Tuple[np.ndarray, dict]: Observation and info.
        """
        super().reset(seed=seed)

        self._setup_data()
        self.episode_steps = 0
        self.total_rewards = 0
        self.buffer_idx = 0
        self._env_state = {
            "demand": self.DEMAND,
            "demand_time": self.DEMAND_TIME,
            "incurred_costs": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            "recurring_costs": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            "production_rates": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            "setup_times": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            "cfgs_status": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            "produced_counts": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            "market_incurring_costs": self.INCUR_COSTS,
            "market_recurring_costs": self.RECUR_COSTS,
            "market_production_rates": self.PRODN_RATES,
            "market_setup_times": self.SETUP_TIMES,
        }
        # static state are used for stochastic operations
        self._static_state = {
            "recurring_costs": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
            "production_rates": np.zeros(self.BUFFER_SIZE, dtype=np.float32),
        }

        if self.RENDER_MODE == "human":
            sns.set()
            self.fig, self.axes = plt.subplots(6, 2, figsize=(10, 7))
            self.fig.suptitle("Manufacturing Environment")
            self._render_frame(action=-1, reward=0)

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Performs one step in environment.
        Args:
            action (int): Action index.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]:
                Observation, reward, terminated, truncated, info.
        """
        assert 0 <= action <= self.BUFFER_SIZE, "Invalid action"

        if self.buffer_idx < self.BUFFER_SIZE:
            if action < self.NUM_CFGS:
                info = {"msg": f"Decision step. Purchase cfg: {action}"}
                reward = self.buy_cfg(cfg_id=action)
            else:
                info = {"msg": "Continuing production"}
                reward = self.continue_production()

        # check for termination after performing action
        terminated, rw, info2 = self._check_for_termination()

        if terminated:
            reward += rw
            info = info2
        elif self.buffer_idx == self.BUFFER_SIZE:
            # check if the action was a final buy action
            # continue simulation until the end and accumulate rewards
            while not terminated:
                reward += self.continue_production()
                terminated, r, info = self._check_for_termination()
                reward += r

        # update total rewards
        self.total_rewards += reward

        # render
        if self.RENDER_MODE == "human":
            self._render_frame(action=action, reward=reward)
            if terminated:
                plt.show(block=True)
                plt.close("all")

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            info,
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
        # update inucrred costs
        self._env_state["incurred_costs"][self.buffer_idx] = self._env_state[
            "market_incurring_costs"
        ][cfg_id]

        # update recurring costs
        self._env_state["recurring_costs"][self.buffer_idx] = self._env_state[
            "market_recurring_costs"
        ][cfg_id]
        self._static_state["recurring_costs"][self.buffer_idx] = self._env_state[
            "market_recurring_costs"
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

        return reward

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
        self._env_state["demand"] = self.DEMAND - np.sum(
            self._env_state["produced_counts"].astype(int)
        )
        self._env_state["demand_time"] -= 1

        # update stochastic parameters
        if self.STOCHASTIC:
            self._imitate_market_uncertainties()
            self._imitate_production_uncertainties()

        return reward

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
        obs_dict["incurred_costs"] = obs_vec[start : start + self.BUFFER_SIZE]

        start += self.BUFFER_SIZE
        obs_dict["recurring_costs"] = obs_vec[start : start + self.BUFFER_SIZE]

        start += self.BUFFER_SIZE
        obs_dict["production_rates"] = obs_vec[start : start + self.BUFFER_SIZE]

        start += self.BUFFER_SIZE
        obs_dict["setup_times"] = obs_vec[start : start + self.BUFFER_SIZE]

        start += self.BUFFER_SIZE
        obs_dict["cfgs_status"] = obs_vec[start : start + self.BUFFER_SIZE]

        start += self.BUFFER_SIZE
        obs_dict["produced_counts"] = obs_vec[start : start + self.BUFFER_SIZE]

        start += self.BUFFER_SIZE
        obs_dict["market_incurring_costs"] = obs_vec[start : start + self.NUM_CFGS]

        start += self.NUM_CFGS
        obs_dict["market_recurring_costs"] = obs_vec[start : start + self.NUM_CFGS]

        start += self.NUM_CFGS
        obs_dict["market_production_rates"] = obs_vec[start : start + self.NUM_CFGS]

        start += self.NUM_CFGS
        obs_dict["market_setup_times"] = obs_vec[start : start + self.NUM_CFGS]

        return obs_dict

    def _check_for_termination(self) -> Tuple[bool, float, dict]:
        """Checks whether evironment is terminated or not.

        Returns:
            Tuple[bool, float, dict]: Terminated, reward, info.
        """
        if (self._env_state["demand"] > 0) and (
            (self._env_state["demand_time"] <= 0)
            or (self.episode_steps >= self.MAX_EPISODE_STEPS)
        ):
            # demand was not satisfied within given time limits
            info = {"msg": "Demand was not satisfied."}
            reward = -1.0 * self.PENALTY_K * self._env_state["demand"]
            terminated = True
        elif (
            (self._env_state["demand"] <= 0)
            and (self._env_state["demand_time"] >= 0)
            and (self.episode_steps <= self.MAX_EPISODE_STEPS)
        ):
            info = {"msg": "Demand is satisfied"}
            reward = 0
            terminated = True
        else:
            info = {}
            reward = 0
            terminated = False

        return terminated, reward, info

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
            low=self.INCUR_COSTS - 0.1 * self.INCUR_COSTS,
            high=self.INCUR_COSTS + 0.1 * self.INCUR_COSTS,
        )

        # recurring costs
        self._env_state["market_recurring_costs"] = np.random.uniform(
            low=self.RECUR_COSTS - 0.1 * self.RECUR_COSTS,
            high=self.RECUR_COSTS + 0.1 * self.RECUR_COSTS,
        )

        # production rates
        self._env_state["market_production_rates"] = np.random.uniform(
            low=self.PRODN_RATES - 0.1 * self.PRODN_RATES,
            high=self.PRODN_RATES + 0.1 * self.PRODN_RATES,
        )

        # setup times
        self._env_state["market_setup_times"] = np.random.uniform(
            low=self.SETUP_TIMES - 0.1 * self.SETUP_TIMES,
            high=self.SETUP_TIMES + 0.1 * self.SETUP_TIMES,
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

    def _check_problem_feasibility(self) -> bool:
        """Checks whether the problem is feasible.
        The problem is not feasible if the solution does not exist
        using the manufacturing configuration with highes capacity.

        Returns:
            bool: Feasibility of the problem.
        """
        idx = np.argmax(self.PRODN_RATES)
        return (self.DEMAND_TIME - self.SETUP_TIMES[idx]) * self.PRODN_RATES[
            idx
        ] * self.BUFFER_SIZE > self.DEMAND

    def _setup_data(self):
        """Sets up the data."""
        if self.DATA_FILE is None:
            # print("ENV IS INITIALIZED IN GENERAL MODE".center(150, "!"))

            feasible_problem = False
            count = 0
            while not feasible_problem:
                count += 1
                # print(f"Trying to generate a feasible problem [{count}]")

                self.DEMAND = np.random.randint(1000, 10000)
                self.DEMAND_TIME = np.random.randint(100, 1000)
                self.MAX_EPISODE_STEPS = self.BUFFER_SIZE + self.DEMAND_TIME

                self.INCUR_COSTS = np.random.randint(500, 10000, self.NUM_CFGS)
                self.RECUR_COSTS = np.random.randint(10, 100, self.NUM_CFGS)
                self.PRODN_RATES = np.random.uniform(0.5, 10.0, self.NUM_CFGS)
                self.SETUP_TIMES = np.random.uniform(1.0, 10.0, self.NUM_CFGS)

                feasible_problem = self._check_problem_feasibility()
        else:
            with open(self.DATA_FILE, "r") as f:
                data = json.load(f)

            assert self.BUFFER_SIZE == data["buffer_size"] and self.NUM_CFGS == len(
                data["configurations"]
            ), "MISMATCH in BUFFER_SIZE and/or NUM_CFGS"

            self.DEMAND = data["demand"]
            self.DEMAND_TIME = data["demand_time"]
            self.MAX_EPISODE_STEPS = self.BUFFER_SIZE + self.DEMAND_TIME
            self.INCUR_COSTS = np.array([], dtype=np.float32)
            self.RECUR_COSTS = np.array([], dtype=np.float32)
            self.PRODN_RATES = np.array([], dtype=np.float32)
            self.SETUP_TIMES = np.array([], dtype=np.float32)
            for v in data["configurations"].values():
                self.INCUR_COSTS = np.append(self.INCUR_COSTS, v["incurring_cost"])
                self.RECUR_COSTS = np.append(self.RECUR_COSTS, v["recurring_cost"])
                self.PRODN_RATES = np.append(self.PRODN_RATES, v["production_rate"])
                self.SETUP_TIMES = np.append(self.SETUP_TIMES, v["setup_time"])

            assert (
                self._check_problem_feasibility()
            ), "Infeasible. Demand will not be satisfied even in the best case."

        # calculate the penalty K
        # K = max. possible incur. cost + max. possible recur. cost
        # max. possible incur.cost = purchasing the most expensive and filling buffer
        # max. possible recur. cost = running the most recur. cost equipment
        self.PENALTY_K = (
            self.INCUR_COSTS.max() + self.MAX_EPISODE_STEPS * self.RECUR_COSTS.max()
        ) * self.BUFFER_SIZE

    def _render_frame(self, **kwargs):
        """Renders one step of environment."""
        for ax in self.axes.flatten():
            ax.clear()

        buffer_idxs = [f"B{i}" for i in range(self.BUFFER_SIZE)]
        market_cfgs = [f"Mfg{i}" for i in range(self.NUM_CFGS)]
        palette = sns.color_palette()

        # remaining demand and time
        text_kwargs = dict(ha="center", va="center", fontsize=12)
        self.axes[0, 0].text(
            0.5,
            0.5,
            f"Remaining demand: {self._env_state['demand']}. "
            f"Remaining time: {self._env_state['demand_time']}",
            **text_kwargs,
        )
        self.axes[0, 0].set_yticklabels([])
        self.axes[0, 0].set_xticklabels([])
        self.axes[0, 0].grid(False)

        # cost
        text_kwargs = dict(ha="center", va="center", fontsize=12)
        action = kwargs["action"]
        reward = kwargs["reward"]
        self.axes[1, 0].text(
            0.5,
            0.5,
            f"Action: {action}. Step reward: {reward:.2f}. "
            f" Total rewards: {self.total_rewards:.2f}",
            **text_kwargs,
        )
        self.axes[1, 0].set_yticklabels([])
        self.axes[1, 0].set_xticklabels([])
        self.axes[1, 0].grid(False)

        # plot market incurring costs
        self.axes[2, 0].bar(
            market_cfgs, self._env_state["market_incurring_costs"], color=palette[3]
        )
        self.axes[2, 0].set_ylabel("£")
        self.axes[2, 0].set_xticklabels([])
        self.axes[2, 0].set_title("Incurring costs (market)")

        # plot market recurring costs
        self.axes[3, 0].bar(
            market_cfgs, self._env_state["market_recurring_costs"], color=palette[4]
        )
        self.axes[3, 0].set_ylabel("kWh")
        self.axes[3, 0].set_xticklabels([])
        self.axes[3, 0].set_title("Recurring costs (market)")

        # plot production rates
        self.axes[4, 0].bar(
            market_cfgs, self._env_state["market_production_rates"], color=palette[5]
        )
        self.axes[4, 0].set_ylabel("p/h")
        self.axes[4, 0].set_xticklabels([])
        self.axes[4, 0].set_title("Production rates (market)")

        # plot setup times
        self.axes[5, 0].bar(
            market_cfgs, self._env_state["market_setup_times"], color=palette[6]
        )
        self.axes[5, 0].set_ylabel("h")
        self.axes[5, 0].set_title("Setup times (market)")
        self.axes[5, 0].set_xlabel("Available configs.")

        # plot cfgs statuses
        progress_colors = [
            palette[2] if p == 1 else palette[1] for p in self._env_state["cfgs_status"]
        ]
        self.axes[0, 1].bar(
            buffer_idxs, self._env_state["cfgs_status"] * 100, color=progress_colors
        )
        self.axes[0, 1].set_ylabel("%")
        self.axes[0, 1].set_ylim([0, 100])
        self.axes[0, 1].set_xticklabels([])
        self.axes[0, 1].set_title("Configurations status (buffer)")

        # plot produced counts
        self.axes[1, 1].bar(
            buffer_idxs, self._env_state["produced_counts"], color=palette[0]
        )
        self.axes[1, 1].set_ylabel("unit")
        self.axes[1, 1].set_ylim(bottom=0)
        self.axes[1, 1].set_xticklabels([])
        self.axes[1, 1].set_title("Production (buffer)")

        # plot incurred costs
        self.axes[2, 1].bar(
            buffer_idxs, self._env_state["incurred_costs"], color=palette[3]
        )
        self.axes[2, 1].set_ylabel("£")
        self.axes[2, 1].set_xticklabels([])
        self.axes[2, 1].set_title("Incurred costs (buffer)")

        # plot recurring costs
        self.axes[3, 1].bar(
            buffer_idxs, self._env_state["recurring_costs"], color=palette[4]
        )
        self.axes[3, 1].set_ylabel("kWh")
        self.axes[3, 1].set_xticklabels([])
        self.axes[3, 1].set_title("Recurring costs (buffer)")

        # plot production rates
        self.axes[4, 1].bar(
            buffer_idxs, self._env_state["production_rates"], color=palette[5]
        )
        self.axes[4, 1].set_ylabel("p/h")
        self.axes[4, 1].set_xticklabels([])
        self.axes[4, 1].set_title("Production rates (buffer)")

        # plot setup times
        self.axes[5, 1].bar(
            buffer_idxs, self._env_state["setup_times"], color=palette[6]
        )
        self.axes[5, 1].set_ylabel("h")
        self.axes[5, 1].set_title("Setup times (buffer)")
        self.axes[5, 1].set_xlabel("Available buffer")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.tight_layout()
        plt.pause(0.1)
