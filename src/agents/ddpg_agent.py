import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.networks import Actor, Critic
from src.agents.replay_buffer import ReplayBuffer


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient agent.

    State dimension: 175
    Action dimension: 29

    Actions are constrained to [-1, 1].
    """

    def __init__(
        self,
        state_dim=175,
        action_dim=29,
        hidden_dim=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100_000,
        batch_size=64,
        noise_std=0.1,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_std = noise_std

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        # --------------------------------------------------------------
        # Networks
        # --------------------------------------------------------------

        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # Target networks start as exact copies.
        self.target_actor = copy.deepcopy(self.actor).to(
            self.device
        )

        self.target_critic = copy.deepcopy(self.critic).to(
            self.device
        )

        # Target networks are not directly optimized.
        for parameter in self.target_actor.parameters():
            parameter.requires_grad = False

        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False

        # --------------------------------------------------------------
        # Optimizers
        # --------------------------------------------------------------

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        # --------------------------------------------------------------
        # Replay buffer
        # --------------------------------------------------------------

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity
        )

    # ------------------------------------------------------------------
    # ACTION
    # ------------------------------------------------------------------

    def select_action(
        self,
        state,
        add_noise=True,
    ):
        """
        Select an action from the Actor.

        Returns:
            numpy array with shape (action_dim,)
            and values in [-1, 1].
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        if state.shape != (self.state_dim,):
            raise ValueError(
                f"Expected state shape "
                f"({self.state_dim},), "
                f"got {state.shape}"
            )

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.actor.eval()

        with torch.no_grad():
            action = self.actor(state_tensor)

        self.actor.train()

        action = action.squeeze(0).cpu().numpy()

        if add_noise:
            noise = np.random.normal(
                loc=0.0,
                scale=self.noise_std,
                size=self.action_dim,
            )

            action += noise

        return np.clip(
            action,
            -1.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # REPLAY
    # ------------------------------------------------------------------

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
        )

    # ------------------------------------------------------------------
    # LEARN
    # ------------------------------------------------------------------

    def update(self):
        """
        Perform one DDPG update.

        Returns:
            dictionary containing critic and actor losses.

        Returns None if the replay buffer does not yet contain
        enough samples.
        """

        if len(self.replay_buffer) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.as_tensor(
            states,
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.float32,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        next_states = torch.as_tensor(
            next_states,
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        # --------------------------------------------------------------
        # Critic update
        # --------------------------------------------------------------

        with torch.no_grad():
            next_actions = self.target_actor(
                next_states
            )

            target_q = self.target_critic(
                next_states,
                next_actions,
            )

            target = rewards + (
                self.gamma
                * (1.0 - dones)
                * target_q
            )

        current_q = self.critic(
            states,
            actions,
        )

        critic_loss = nn.functional.mse_loss(
            current_q,
            target,
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=1.0,
        )

        self.critic_optimizer.step()

        # --------------------------------------------------------------
        # Actor update
        # --------------------------------------------------------------

        predicted_actions = self.actor(
            states
        )

        actor_loss = -self.critic(
            states,
            predicted_actions,
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=1.0,
        )

        self.actor_optimizer.step()

        # --------------------------------------------------------------
        # Target network update
        # --------------------------------------------------------------

        self._soft_update(
            self.actor,
            self.target_actor,
        )

        self._soft_update(
            self.critic,
            self.target_critic,
        )

        return {
            "critic_loss": float(
                critic_loss.item()
            ),
            "actor_loss": float(
                actor_loss.item()
            ),
        }

    # ------------------------------------------------------------------
    # SOFT UPDATE
    # ------------------------------------------------------------------

    def _soft_update(
        self,
        source,
        target,
    ):
        """
        target = tau * source + (1 - tau) * target
        """

        for target_param, source_param in zip(
            target.parameters(),
            source.parameters(),
        ):
            target_param.data.copy_(
                self.tau * source_param.data
                + (1.0 - self.tau)
                * target_param.data
            )

    # ------------------------------------------------------------------
    # SAVE / LOAD
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": (
                    self.actor_optimizer.state_dict()
                ),
                "critic_optimizer": (
                    self.critic_optimizer.state_dict()
                ),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.actor.load_state_dict(
            checkpoint["actor"]
        )

        self.critic.load_state_dict(
            checkpoint["critic"]
        )

        self.target_actor.load_state_dict(
            checkpoint["target_actor"]
        )

        self.target_critic.load_state_dict(
            checkpoint["target_critic"]
        )

        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )

        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )