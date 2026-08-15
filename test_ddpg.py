import numpy as np

from src.agents.ddpg_agent import DDPGAgent
from src.environment.trading_env import TradingEnvironment


env = TradingEnvironment()

agent = DDPGAgent(
    state_dim=175,
    action_dim=29,
    batch_size=32,
)

state = env.reset()

total_reward = 0.0
updates = 0

print("Initial state shape:", state.shape)
print("Initial portfolio:", env.portfolio_value)

for step in range(100):

    action = agent.select_action(
        state,
        add_noise=True,
    )

    next_state, reward, done, info = env.step(
        action
    )

    agent.remember(
        state,
        action,
        reward,
        next_state,
        done,
    )

    loss = agent.update()

    if loss is not None:
        updates += 1

    total_reward += reward
    state = next_state

    if step < 5:
        print(
            f"Step {step + 1}: "
            f"reward={reward:.2f}, "
            f"portfolio={info['portfolio_value']:.2f}, "
            f"shares={info['holdings'].sum():.0f}"
        )

    if done:
        break


print()
print("=== DDPG SMOKE TEST ===")
print("Steps:", step + 1)
print("Replay buffer:", len(agent.replay_buffer))
print("Updates:", updates)
print("Total reward:", total_reward)
print("Final portfolio:", env.portfolio_value)
print("Final holdings:", env.holdings.sum())
print("Last actor loss:", loss["actor_loss"] if loss else None)
print("Last critic loss:", loss["critic_loss"] if loss else None)
print("PASS")