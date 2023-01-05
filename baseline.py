import gymnasium as gym

env = gym.make("mfgrl:mfgrl/MfgEnv-v0")

obs, info = env.reset(seed=42)
total_reward = 0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(total_reward)
