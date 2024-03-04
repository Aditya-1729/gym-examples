"""
Gym Cartpole-v1.0 Environment with modified reward function to include energy penalty
"""
import math
from typing import Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils, CartPoleEnv
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.vector import VectorEnv
from gymnasium.vector.utils import batch_space


class CustomCartPoleEnv(CartPoleEnv):
    """
    ## Description

    An implementation of an energy efficient cartPole environment. The goal of this environment is to keep the pole upright.
    Inherits from the `CartPoleEnv` class and overrides the `step` method to include a penalty for the total energy of the system.

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken
    Additionally, a penalty of `1*total_energy` is applied to the reward. Only kinetaic energy of the system is penalized.
    The ideal state of the cartpole:
    -  Maximizes the potential energy of pole_mass*gravity*length/2 and 
    -  Minizes the kinetic energy of 0.5*cart_mass*cart_velocity^2 + 0.5*pole_mass*cart_velocity^2 + 0.5*pole_inertia*pole_angular_velocity^2
    In random runs, the observed maximum kinetic energy was 1.7.  As it is comparable in magnitude with the default positive reward of 1,
      a multiplier of 1 is used as default and the total reward is formulated as `1 - total_energy`.
      
    ## Arguments

    ```python
    import gymnasium as gym
    import gym_examples
    env = gym.make("gym_examples/EfficientCartpole")
    ```
    """

    def __init__(self, render_mode: Optional[str] = None, cfg: DictConfig = None):
        super().__init__(render_mode)

        self.cfg=cfg
        self.energy_mul=cfg.energy_mul

        #modify the action space to include a no action option
        self.action_space = spaces.Discrete(3)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        
        #Empty info dict at the start of each step
        info = {}
        
        x, x_dot, theta, theta_dot = self.state

        #action 1 is to push the cart to the right, action 0 is to do nothing, action 2 is to
        if action == 1:
            force = self.force_mag
        elif action == 0:
            force = 0  
        else:
            force = -self.force_mag
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass


        #Energy of the system [https://courses.ece.ucsb.edu/ECE594/594D_W10Byl/hw/cartpole_eom.pdf]
        self.kinetic_energy = 0.5 * self.total_mass*x_dot**2 + 0.5 * self.polemass_length*self.length*theta_dot**2 - self.polemass_length*x_dot*theta_dot*math.cos(theta)
        # potential energy is ignored in the penalty as the potential energy is maximized when the pole tends to be upright
        self.total_energy = self.kinetic_energy 
        self.energy_penalty = self.energy_mul*self.total_energy 
        info['energy'] = self.total_energy

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0 - self.total_energy if self.cfg.energy_penalty else 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0 - self.total_energy if self.cfg.energy_penalty else 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, info



