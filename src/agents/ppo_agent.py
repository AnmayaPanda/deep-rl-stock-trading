import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
    """
    PPO Actor-Critic network.

    Input:
        state: 175 dimensions

    Outputs:
        action_mean: 29 continuous actions
        state_value: scalar V(s)
    """

    def __init__(
        self,
        state_dim=175,
        action_dim=29,
        hidden_dim=256,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.critic = nn.Linear(
            hidden_dim,
            1,
        )

        # Learnable standard deviation for continuous actions.
        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

    def forward(self, state):
        features = self.shared(state)

        action_mean = self.actor(features)

        value = self.critic(features)

        return action_mean, value


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    State dimension: 175
    Action dimension: 29

    Continuous actions in [-1, 1].
    """

    def __init__(
        self,
        state_dim=175,
        action_dim=29,
        hidden_dim=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=10,
        batch_size=64,
        device=None,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.epochs = epochs
        self.batch_size = batch_size

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.network = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=learning_rate,
        )

    # --------------------------------------------------------------
    # ACTION
    # --------------------------------------------------------------

    def select_action(
        self,
        state,
        deterministic=False,
    ):
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

        self.network.eval()

        with torch.no_grad():

            action_mean, value = self.network(
                state_tensor
            )

            std = torch.exp(
                self.network.log_std
            )

            distribution = torch.distributions.Normal(
                action_mean,
                std,
            )

            if deterministic:
                raw_action = action_mean
            else:
                raw_action = distribution.sample()

            action = torch.tanh(
                raw_action
            )

            # Correct log probability for tanh squashing.
            log_prob = distribution.log_prob(
                raw_action
            ) - torch.log(
                1.0 - action.pow(2) + 1e-6
            )

            log_prob = log_prob.sum(
                dim=-1
            )

        self.network.train()

        return (
            action.squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32),

            float(log_prob.item()),

            float(value.item()),
        )

    # --------------------------------------------------------------
    # LOG PROBABILITY
    # --------------------------------------------------------------

    def evaluate_actions(
        self,
        states,
        actions,
    ):

        action_mean, values = self.network(
            states
        )

        std = torch.exp(
            self.network.log_std
        )

        distribution = torch.distributions.Normal(
            action_mean,
            std,
        )

        # Numerical protection before atanh.
        actions = torch.clamp(
            actions,
            -0.999999,
            0.999999,
        )

        raw_actions = torch.atanh(
            actions
        )

        log_probs = distribution.log_prob(
            raw_actions
        ) - torch.log(
            1.0 - actions.pow(2) + 1e-6
        )

        log_probs = log_probs.sum(
            dim=-1
        )

        entropy = distribution.entropy().sum(
            dim=-1
        )

        return (
            log_probs,
            entropy,
            values.squeeze(-1),
        )

    # --------------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------------

    def update(
        self,
        states,
        actions,
        old_log_probs,
        returns,
        advantages,
    ):

        states = torch.as_tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.as_tensor(
            np.asarray(actions),
            dtype=torch.float32,
            device=self.device,
        )

        old_log_probs = torch.as_tensor(
            np.asarray(old_log_probs),
            dtype=torch.float32,
            device=self.device,
        )

        returns = torch.as_tensor(
            np.asarray(returns),
            dtype=torch.float32,
            device=self.device,
        )

        advantages = torch.as_tensor(
            np.asarray(advantages),
            dtype=torch.float32,
            device=self.device,
        )

        # Normalize advantages.
        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        dataset_size = states.shape[0]

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        update_count = 0

        for _ in range(self.epochs):

            indices = np.random.permutation(
                dataset_size
            )

            for start in range(
                0,
                dataset_size,
                self.batch_size,
            ):

                batch_indices = indices[
                    start:
                    start + self.batch_size
                ]

                batch_states = states[
                    batch_indices
                ]

                batch_actions = actions[
                    batch_indices
                ]

                batch_old_log_probs = (
                    old_log_probs[
                        batch_indices
                    ]
                )

                batch_returns = returns[
                    batch_indices
                ]

                batch_advantages = advantages[
                    batch_indices
                ]

                (
                    new_log_probs,
                    entropy,
                    values,
                ) = self.evaluate_actions(
                    batch_states,
                    batch_actions,
                )

                ratio = torch.exp(
                    new_log_probs
                    - batch_old_log_probs
                )

                unclipped = (
                    ratio
                    * batch_advantages
                )

                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * batch_advantages

                actor_loss = -torch.min(
                    unclipped,
                    clipped,
                ).mean()

                critic_loss = nn.functional.mse_loss(
                    values,
                    batch_returns,
                )

                entropy_loss = entropy.mean()

                loss = (
                    actor_loss
                    + self.value_coef
                    * critic_loss
                    - self.entropy_coef
                    * entropy_loss
                )

                self.optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    max_norm=1.0,
                )

                self.optimizer.step()

                total_actor_loss += (
                    actor_loss.item()
                )

                total_critic_loss += (
                    critic_loss.item()
                )

                total_entropy += (
                    entropy_loss.item()
                )

                update_count += 1

        return {
            "actor_loss":
                total_actor_loss
                / max(update_count, 1),

            "critic_loss":
                total_critic_loss
                / max(update_count, 1),

            "entropy":
                total_entropy
                / max(update_count, 1),
        }

    # --------------------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------------------

    def save(self, path):

        torch.save(
            {
                "network":
                    self.network.state_dict(),

                "optimizer":
                    self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.network.load_state_dict(
            checkpoint["network"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )