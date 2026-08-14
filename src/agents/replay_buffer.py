import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for DDPG.

    Stores:
        state
        action
        reward
        next_state
        done
    """

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                float(done),
            )
        )

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"Not enough samples in buffer: "
                f"{len(self.buffer)} < {batch_size}"
            )

        batch = random.sample(
            self.buffer,
            batch_size,
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        return (
            np.stack(states),
            np.stack(actions),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()