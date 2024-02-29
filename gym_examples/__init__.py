from gymnasium.envs.registration import register

register(
    id="gym_examples/EfficientCartpole",
    entry_point="gym_examples.envs:EffCartPoleEnv",
    max_episode_steps=500
)