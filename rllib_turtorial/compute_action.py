# Note: `gymnasium` (not `gym`) will be **the** API supported by RLlib from Ray 2.3 on.
import gymnasium as gym

from ray.rllib.algorithms.ppo import PPOConfig, PPO

env_name = "CartPole-v1"
env = gym.make(env_name)

# untrained algo
# algo = PPOConfig().environment(env_name).build()

# trianed algo
checkpoint_path = "/root/ray_results/PPO_2024-03-23_19-22-21/PPO_CartPole-v1_9d6ee_00001_1_lr=0.0010_2024-03-23_19-22-21/checkpoint_000000"
algo = PPO.from_checkpoint(checkpoint_path)

episode_reward = 0
terminated = truncated = False


obs, info = env.reset()

while not terminated and not truncated:
    action = algo.compute_single_action(obs)
 
    obs, reward, terminated, truncated, info = env.step(action)

    episode_reward += reward

print(f'episode_reward: {episode_reward}')