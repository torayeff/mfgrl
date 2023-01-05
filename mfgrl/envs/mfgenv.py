import gymnasium as gym


class MfgEnv(gym.Env):
    def __init__(self):
        self.observation_space = None
        self.action_space = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = None
        reward = 0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def step(self, action):
        obs = None
        reward = 0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
