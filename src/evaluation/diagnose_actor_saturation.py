import numpy as np
import torch

from src.agents.ddpg_agent import DDPGAgent
from src.environment.trading_env_weighted import TradingEnvironment


MODEL_PATH = "ddpg_train_2009_2015_weighted.pt"


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

agent.load(MODEL_PATH)

state = env.reset()

all_actions = []
all_pretanh = []

done = False

while not done:

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():

        x = state_tensor

        x = agent.actor.network[0](x)
        x = agent.actor.network[1](x)

        x = agent.actor.network[2](x)
        x = agent.actor.network[3](x)

        pre_tanh = agent.actor.network[4](x)

        action = torch.tanh(pre_tanh)

    all_pretanh.append(
        pre_tanh.squeeze(0).numpy()
    )

    all_actions.append(
        action.squeeze(0).numpy()
    )

    state, reward, done, info = env.step(
        action.squeeze(0).numpy()
    )


all_actions = np.asarray(all_actions)
all_pretanh = np.asarray(all_pretanh)


print("=== ACTOR SATURATION DIAGNOSTICS ===")

print("Validation steps:", len(all_actions))
print("Stocks:", all_actions.shape[1])

print()
print("=== ACTIONS ===")

print(
    "Mean:",
    all_actions.mean()
)

print(
    "Std:",
    all_actions.std()
)

print(
    "Mean absolute:",
    np.abs(all_actions).mean()
)

print(
    ">= +0.95:",
    (all_actions >= 0.95).mean() * 100,
    "%",
)

print(
    "<= -0.95:",
    (all_actions <= -0.95).mean() * 100,
    "%",
)

print(
    "Within (-0.95,+0.95):",
    (
        (all_actions > -0.95)
        & (all_actions < 0.95)
    ).mean() * 100,
    "%",
)

print()
print("=== PRE-TANH ===")

print(
    "Mean:",
    all_pretanh.mean()
)

print(
    "Std:",
    all_pretanh.std()
)

print(
    "Mean absolute:",
    np.abs(all_pretanh).mean()
)

print(
    "|pre-tanh| >= 3:",
    (np.abs(all_pretanh) >= 3).mean() * 100,
    "%",
)

print(
    "|pre-tanh| >= 4:",
    (np.abs(all_pretanh) >= 4).mean() * 100,
    "%",
)

print(
    "|pre-tanh| >= 5:",
    (np.abs(all_pretanh) >= 5).mean() * 100,
    "%",
)