import gym

print(gym.__version__)
env = gym.make("CartPole-v1")

x = env.reset()
print(x)
print(env.action_space.sample(), env.action_space.n)
print(env.action_space)
print(env.observation_space)
print()
print(env.step(0))