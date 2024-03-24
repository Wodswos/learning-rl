from ray.rllib.algorithms.dqn import DQNConfig

algo = DQNConfig().environment(env="CartPole-v1").build()

# Get weights of the default local policy
policy = algo.get_policy()
# print(len(policy.get_weights()))
[print(key) for key in policy.get_weights().keys()]
# print(policy.get_weights())



# Same as above
# print(
#     "local worker default policy's weights: "
#     f'{algo.workers.local_worker().policy_map["default_policy"].get_weights()}'
# )

# # Get list of weights of each worker, including remote replicas
# algo.workers.foreach_worker(lambda worker: worker.get_policy().get_weights())

# # Same as above, but with index.
# algo.workers.foreach_worker_with_id(
#     lambda _id, worker: worker.get_policy().get_weights()
# )