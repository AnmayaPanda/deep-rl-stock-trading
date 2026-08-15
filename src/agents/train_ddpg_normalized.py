import numpy as np

from src.agents.ddpg_agent import DDPGAgent
from src.environment.trading_env_normalized import TradingEnvironment


def train(
    episodes=1,
    max_steps=None,
    batch_size=32,
):
    env = TradingEnvironment(
        start_date="2009-01-01",
        end_date="2015-09-30",
        turbulence_threshold=71.1,
    )

    agent = DDPGAgent(
        state_dim=175,
        action_dim=29,
        batch_size=batch_size,
    )

    print("=== DDPG TRAINING ===")
    print("Device:", agent.device)
    print("Stocks:", env.n_stocks)
    print("Trading days:", env.n_steps)
    print("State dimension:", agent.state_dim)
    print("Action dimension:", agent.action_dim)
    print()

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0.0
        update_count = 0

        steps = (
            env.n_steps - 1
            if max_steps is None
            else min(max_steps, env.n_steps - 1)
        )

        for step in range(steps):

            # ----------------------------------------------------------
            # Select action
            # ----------------------------------------------------------

            action = agent.select_action(
                state,
                add_noise=True,
            )

            # ----------------------------------------------------------
            # Environment step
            # ----------------------------------------------------------

            next_state, reward, done, info = env.step(
                action
            )

            # ----------------------------------------------------------
            # Store transition
            # ----------------------------------------------------------

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            # ----------------------------------------------------------
            # Update DDPG
            # ----------------------------------------------------------

            loss = agent.update()

            if loss is not None:
                update_count += 1

            total_reward += reward
            state = next_state

            # ----------------------------------------------------------
            # Progress
            # ----------------------------------------------------------

            if step < 5 or (step + 1) % 100 == 0:
                print(
                    f"Episode {episode + 1} | "
                    f"Step {step + 1}/{steps} | "
                    f"Reward: {reward:.2f} | "
                    f"Portfolio: "
                    f"{info['portfolio_value']:.2f} | "
                    f"Shares: "
                    f"{info['holdings'].sum():.0f}"
                )

            if done:
                break

        # --------------------------------------------------------------
        # Episode summary
        # --------------------------------------------------------------

        print()
        print(
            f"Episode {episode + 1} complete"
        )
        print(
            f"Steps: {step + 1}"
        )
        print(
            f"Total reward: {total_reward:.2f}"
        )
        print(
            f"Final portfolio: "
            f"{env.portfolio_value:.2f}"
        )
        print(
            f"Total transaction costs: "
            f"{env.total_transaction_costs:.2f}"
        )
        print(
            f"Replay buffer: "
            f"{len(agent.replay_buffer)}"
        )
        print(
            f"Updates: {update_count}"
        )

        if loss is not None:
            print(
                f"Actor loss: "
                f"{loss['actor_loss']:.6f}"
            )
            print(
                f"Critic loss: "
                f"{loss['critic_loss']:.6f}"
            )

        print()

    return env, agent


if __name__ == "__main__":

    env, agent = train(
        episodes=1,
        max_steps=None,
        batch_size=32,
    )

    agent.save(
        "ddpg_train_2009_2015_normalized.pt"
    )

    print()
    print("Model saved: ddpg_train_2009_2015_normalized.pt")
