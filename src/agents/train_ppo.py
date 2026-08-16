import numpy as np

from src.agents.ppo_agent import PPOAgent
from src.environment.trading_env_weighted import TradingEnvironment


TRAIN_START_DATE = "2009-01-01"
TRAIN_END_DATE = "2015-09-30"

MODEL_PATH = (
    "ppo_train_2009_2015_weighted_smoke.pt"
)


def calculate_gae(
    rewards,
    values,
    next_value,
    dones,
    gamma,
    gae_lambda,
):
    """
    Calculate Generalized Advantage Estimation (GAE).

    advantages[t] =
        delta[t]
        + gamma * lambda * (1 - done[t]) * advantages[t+1]

    where:

        delta[t] =
            reward[t]
            + gamma * next_value[t]
            - value[t]
    """

    rewards = np.asarray(
        rewards,
        dtype=np.float32,
    )

    values = np.asarray(
        values,
        dtype=np.float32,
    )

    dones = np.asarray(
        dones,
        dtype=np.float32,
    )

    advantages = np.zeros_like(
        rewards,
        dtype=np.float32,
    )

    gae = 0.0

    for t in reversed(
        range(len(rewards))
    ):

        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        delta = (
            rewards[t]
            + gamma
            * next_val
            * (1.0 - dones[t])
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * (1.0 - dones[t])
            * gae
        )

        advantages[t] = gae

    returns = (
        advantages + values
    )

    return advantages, returns


def train(
    episodes=2,
    batch_size=64,
):

    env = TradingEnvironment(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        turbulence_threshold=None,
    )

    agent = PPOAgent(
        state_dim=175,
        action_dim=29,
        batch_size=batch_size,
    )

    print("=== PPO TRAINING ===")
    print("Device:", agent.device)
    print("Stocks:", env.n_stocks)
    print("Trading days:", env.n_steps)
    print("State dimension:", agent.state_dim)
    print("Action dimension:", agent.action_dim)
    print()

    for episode in range(episodes):

        state = env.reset()

        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        total_reward = 0.0

        steps = env.n_steps - 1

        for step in range(steps):

            # ------------------------------------------------------
            # Select action
            # ------------------------------------------------------

            (
                action,
                log_prob,
                value,
            ) = agent.select_action(
                state,
                deterministic=False,
            )

            # ------------------------------------------------------
            # Environment step
            # ------------------------------------------------------

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            # ------------------------------------------------------
            # Store transition
            # ------------------------------------------------------

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)

            total_reward += reward

            state = next_state

            # ------------------------------------------------------
            # Progress
            # ------------------------------------------------------

            if (
                step < 5
                or (step + 1) % 100 == 0
            ):
                print(
                    f"Episode {episode + 1} | "
                    f"Step {step + 1}/{steps} | "
                    f"Reward: {reward:.4f} | "
                    f"Portfolio: "
                    f"{info['portfolio_value']:.2f}"
                )

            if done:
                break

        # ----------------------------------------------------------
        # Bootstrap value
        # ----------------------------------------------------------

        if dones[-1]:
            next_value = 0.0
        else:
            _, next_value, _ = agent.select_action(
                state,
                deterministic=True,
            )

        # ----------------------------------------------------------
        # GAE
        # ----------------------------------------------------------

        advantages, returns = calculate_gae(
            rewards=rewards,
            values=values,
            next_value=next_value,
            dones=dones,
            gamma=agent.gamma,
            gae_lambda=agent.gae_lambda,
        )

        # ----------------------------------------------------------
        # PPO update
        # ----------------------------------------------------------

        losses = agent.update(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        )

        # ----------------------------------------------------------
        # Episode summary
        # ----------------------------------------------------------

        print()
        print(
            f"Episode {episode + 1} complete"
        )

        print(
            f"Steps: {len(rewards)}"
        )

        print(
            f"Total reward: "
            f"{total_reward:.4f}"
        )

        print(
            f"Final portfolio: "
            f"{env.portfolio_value:.2f}"
        )

        print(
            f"Transaction costs: "
            f"{env.total_transaction_costs:.2f}"
        )

        if losses is not None:

            print(
                f"Actor loss: "
                f"{losses['actor_loss']:.6f}"
            )

            print(
                f"Critic loss: "
                f"{losses['critic_loss']:.6f}"
            )

            print(
                f"Entropy: "
                f"{losses['entropy']:.6f}"
            )

        print()

    return env, agent


if __name__ == "__main__":

    env, agent = train(
        episodes=2,
        batch_size=64,
    )

    agent.save(
        MODEL_PATH
    )

    print(
        f"Model saved: {MODEL_PATH}"
    )