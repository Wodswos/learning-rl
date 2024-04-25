from ray.rllib.algorithms import PPOConfig

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .rollouts(num_rollout_workers=2)
    .resources(num_gpus=0.5, num_gpus_per_worker=0.1)
    .framework("torch")
    .training(model={"fcnet_hiddens": [64, 64]})
)

algo = config.build()

for _ in range(10):
    result = algo.train()
    # print(pretty_print(result))
    print(f'episode_reward_mean: {result["episode_reward_mean"]}')

input()
