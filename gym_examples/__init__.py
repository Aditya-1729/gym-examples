from gymnasium.envs.registration import register

register(
    id="gym_examples/CustomCartpole",
    entry_point="gym_examples.envs:CustomCartPoleEnv",
    max_episode_steps=500
)