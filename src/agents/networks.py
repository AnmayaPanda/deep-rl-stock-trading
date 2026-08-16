import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(
        self,
        state_dim=175,
        action_dim=29,
        hidden_dim=256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):

    def __init__(
        self,
        state_dim=175,
        action_dim=29,
        hidden_dim=256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                state_dim + action_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(self, state, action):

        x = torch.cat(
            [state, action],
            dim=1,
        )

        return self.network(x)