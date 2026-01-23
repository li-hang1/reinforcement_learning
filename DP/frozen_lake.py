import gymnasium as gym
from policy_iteration import PolicyIteration, print_agent
from value_iteration import ValueIteration


env = gym.make("FrozenLake-v1", render_mode="rgb_array")
env = env.unwrapped
obs, info = env.reset()
env.render()


action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])