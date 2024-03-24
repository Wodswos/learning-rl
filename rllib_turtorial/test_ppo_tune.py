import ray
from ray import train, tune
from ray.rllib.algorithms.ppo import PPOConfig

ray.init()

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .rollouts(num_rollout_workers=2)
    # .resources(
    #     num_gpus=0.2,
    #     num_gpus_per_worker=0.1
    # )
    .framework("torch")
    .training(
        model={"fcnet_hiddens": [64, 64]},
        lr=tune.grid_search([0.01, 0.001, 0.0001])
    )
)

tuner = tune.Tuner(
    "PPO",
    run_config=train.RunConfig(
        stop={"episode_reward_mean": 150},
    ),
    param_space=config,
)

results = tuner.fit()

# Get the best result based on a particular metric.
best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
best_checkpoint = best_result.checkpoint

print(best_result)