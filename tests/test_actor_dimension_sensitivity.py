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

    base_q = agent.critic(
        s,
        actor_action,
    ).item()

    print("=== ACTOR DIMENSION SENSITIVITY ===")
    print()
    print("Base Q:", base_q)
    print()

    for i in range(29):

        action_minus = actor_action.clone()
        action_plus = actor_action.clone()

        action_minus[0, i] -= 0.10
        action_plus[0, i] += 0.10

        action_minus = torch.clamp(
            action_minus,
            -1.0,
            1.0,
        )

        action_plus = torch.clamp(
            action_plus,
            -1.0,
            1.0,
        )

        q_minus = agent.critic(
            s,
            action_minus,
        ).item()

        q_plus = agent.critic(
            s,
            action_plus,
        ).item()

        delta_minus = q_minus - base_q
        delta_plus = q_plus - base_q

        print(
            f"Stock {i:2d} | "
            f"Action={actor_action[0, i].item():+.4f} | "
            f"Q(-0.10)={q_minus:.6f} "
            f"(Δ={delta_minus:+.6f}) | "
            f"Q(+0.10)={q_plus:.6f} "
            f"(Δ={delta_plus:+.6f})"
        )