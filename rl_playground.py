import gym
import numpy as np
from ray.air import Checkpoint
from ray.air.config import RunConfig, ScalingConfig
from ray.air.result import Result
from ray.train.rl.rl_predictor import RLPredictor
from ray.train.rl.rl_trainer import RLTrainer
from ray.tune.tuner import Tuner


def train_rl_ppo_online(num_workers: int, use_gpu: bool = False) -> Result:
    print("Starting online training")
    trainer = RLTrainer(
        run_config=RunConfig(stop={"training_iteration": 5}),
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        algorithm="PPO",
        config={
            "env": "CartPole-v1",
            "framework": "torch",
        },
    )
    tuner = Tuner(
        trainer,
        _tuner_kwargs={"checkpoint_at_end": True},
    )
    result = tuner.fit()[0]
    return result


def evaluate_using_checkpoint(checkpoint: Checkpoint, num_episodes) -> list:
    predictor = RLPredictor.from_checkpoint(checkpoint)

    env = gym.make("CartPole-v1")

    rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        reward = 0.0
        done = False
        while not done:
            action = predictor.predict(np.array([obs]))
            obs, r, done, _ = env.step(action[0])
            reward += r
        rewards.append(reward)

    return rewards


result = train_rl_ppo_online(num_workers=2, use_gpu=False)

num_eval_episodes = 3

rewards = evaluate_using_checkpoint(result.checkpoint, num_episodes=num_eval_episodes)
print(f"Average reward over {num_eval_episodes} episodes: " f"{np.mean(rewards)}")
