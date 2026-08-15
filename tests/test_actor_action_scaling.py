import numpy as np
import torch

from src.agents.ddpg_agent import DDPGAgent
from src.environment.trading_env_weighted import TradingEnvironment


env = TradingEnvironment(
    start_date="2015-10-01",
    end_date="2015-12-31",
    turbulence_threshold=71.1,
)

agent = DDPGAgent(
    state_dim=175,
    action_dim=29,
    batch_size=32,
)

agent.load("ddpg_train_2009_2015_weighted.pt")

state = env.reset()

s = torch.as_tensor(
    state,
    dtype=torch.float32,
).unsqueeze(0)

agent.actor.eval()
agent.critic.eval()

with torch.no_grad():

    actor_action = agent.actor(s)

    print("=== ACTOR ACTION SCALING DIAGNOSTIC ===")
    print()

    print(
        "Actor action mean:",
        actor_action.mean().item(),
    )

    print(
        "Actor action abs mean:",
        actor_action.abs().mean().item(),
    )

    print()

    scales = [
        1.00,
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
        0.25,
        0.00,
    ]

    for scale in scales:

        test_action = actor_action * scale

        q_value = agent.critic(
            s,
            test_action,
        ).item()

        print(
            f"Scale {scale:.2f} | "
            f"Q-value: {q_value:.6f} | "
            f"Action abs mean: "
            f"{test_action.abs().mean().item():.6f}"
        )