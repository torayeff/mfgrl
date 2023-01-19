import gymnasium as gym
import numpy as np
import random
import pickle

np.random.seed(4242)
random.seed(4242)

env_config = {
    "data_file": "data.json",
    "scale_costs": True,
    "stochastic": True,
    "render_mode": None,
}
env = gym.make("mfgrl:mfgrl/MfgEnv-v0", env_config=env_config)

N_TRIALS = 1000000
trial = 0

best_total_reward = -10e9
best_policy = None

while trial < N_TRIALS:
    trial += 1
    obs, info = env.reset()

    total_reward = 0
    policy = []
    while True:
        if len(np.where(env.decode_obs(obs)["incurred_costs"] == 0)[0]) == 0:
            # if the buffer is full continue production
            action = env.action_space.n
        else:
            # buy randomly until the buffer is full
            action = np.random.randint(0, env.action_space.n)

        policy.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break

    if trial % 100 == 0 or trial == 1:
        print(f"Trial-{trial}".center(100, "-"))
        print(f"Total reward: {total_reward}")

    if total_reward > best_total_reward:
        best_total_reward = total_reward
        best_policy = policy

print(f"Best total reward: {best_total_reward}")

with open("results_baseline_random.pkl", "wb") as f:
    pickle.dump({"best_total_reward": best_total_reward, "best_policy": best_policy}, f)
