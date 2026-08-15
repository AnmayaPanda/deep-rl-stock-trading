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

state_tensor = torch.as_tensor(
    state,
    dtype=torch.float32,
).unsqueeze(0)

agent.actor.train()
agent.critic.eval()

# ----------------------------------------------------------
# Actor action
# ----------------------------------------------------------

action = agent.actor(state_tensor)

# ----------------------------------------------------------
# Actor objective
# ----------------------------------------------------------

q_value = agent.critic(
    state_tensor,
    action,
)

actor_loss = -q_value.mean()

# ----------------------------------------------------------
# Gradient
# ----------------------------------------------------------

agent.actor.zero_grad()

actor_loss.backward()

# ----------------------------------------------------------
# Collect gradient statistics
# ----------------------------------------------------------

total_norm = 0.0
max_grad = 0.0

for parameter in agent.actor.parameters():

    if parameter.grad is not None:

        grad_norm = parameter.grad.data.norm(2).item()

        total_norm += grad_norm ** 2

        max_grad = max(
            max_grad,
            parameter.grad.data.abs().max().item(),
        )

total_norm = total_norm ** 0.5


print("=== DDPG ACTOR GRADIENT DIAGNOSTIC ===")

print("Actor loss:", actor_loss.item())
print("Critic Q-value:", q_value.item())

print()
print("Total actor gradient norm:", total_norm)
print("Maximum individual gradient:", max_grad)

print()
print("=== GRADIENTS BY LAYER ===")

for name, parameter in agent.actor.named_parameters():

    if parameter.grad is not None:

        print(
            f"{name}: "
            f"mean_abs={parameter.grad.data.abs().mean().item():.8f} "
            f"max_abs={parameter.grad.data.abs().max().item():.8f} "
            f"norm={parameter.grad.data.norm(2).item():.8f}"
        )