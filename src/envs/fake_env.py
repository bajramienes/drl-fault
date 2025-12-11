import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class FakeEnvContinuous(gym.Env):
    """
    Lightweight, continuous-action training environment.

    Used for: PPO, SAC, TD3, DDPG, A2C.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = random.Random(seed)

        # Observation: 6 normalized metrics
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Continuous scalar action in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.max_steps = 20
        self.step_index = 0

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.step_index = 0
        obs = np.zeros((6,), dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.step_index += 1

        # Single scalar action in [-1,1]
        a = float(action[0]) if isinstance(action, (np.ndarray, list, tuple)) else float(action)

        # simple shaping: positive action "tries to scale up", negative "scale down"
        base = np.random.rand(6).astype(np.float32)
        cpu = base[0] + (-0.1 if a > 0.3 else (0.1 if a < -0.3 else 0.0))
        mem = base[1] + (-0.08 if a > 0.3 else (0.08 if a < -0.3 else 0.0))

        obs = base.copy()
        obs[0] = np.clip(cpu, 0.0, 1.0)
        obs[1] = np.clip(mem, 0.0, 1.0)

        # reward: lower cpu+mem better
        reward = - (0.5 * obs[0] + 0.5 * obs[1])

        done = False
        truncated = self.step_index >= self.max_steps

        return obs, float(reward), done, truncated, {}


class FakeEnvDiscrete(gym.Env):
    """
    Lightweight, discrete-action training environment.

    Used for: DQN.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = random.Random(seed)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0=no-op, 1=scale_up, 2=scale_down

        self.max_steps = 20
        self.step_index = 0

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.step_index = 0
        obs = np.zeros((6,), dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        self.step_index += 1

        # base random observation
        obs = np.random.rand(6).astype(np.float32)

        # simple effect of action on cpu/mem
        if action == 1:  # scale_up
            obs[0] = max(0.0, obs[0] - 0.15)
            obs[1] = max(0.0, obs[1] - 0.10)
        elif action == 2:  # scale_down
            obs[0] = min(1.0, obs[0] + 0.15)
            obs[1] = min(1.0, obs[1] + 0.10)

        reward = - (0.5 * obs[0] + 0.5 * obs[1])

        done = False
        truncated = self.step_index >= self.max_steps

        return obs, float(reward), done, truncated, {}
