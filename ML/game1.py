import gymnasium as gym
import numpy as np
import random
# Create environment
env = gym.make("FrozenLake-v1", is_slippery=False)
states = env.observation_space.n
actions = env.action_space.n
# Create Q Table
Q = np.zeros((states, actions))
learning_rate = 0.8
discount_factor = 0.95
episodes = 1000
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        # Random action (Exploration)
        action = random.randint(0, actions - 1)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning Update Formula
        Q[state, action] = Q[state, action] + learning_rate * (
            reward + discount_factor * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

print("Final Q Table:")
print(Q)









# import gymnasium as gym
# import numpy as np
# import random
# env = gym.make("FrozenLake-v1",is_slippery=False)
# states = env.observation_space.n
# actions = env.action_space.n
# Q = np.zeros(states,actions)
# learning_rate = 0.8
# discount_factor = 0.95
# episodes = 1000
# for episode in range(episodes):
#     state,_ = env.reset()
#     done = False

#     while not done:
#         action = random.randint(0, actions-1)
#         next_state ,ResourceWarning,terminated,truncated,_ = env.step(action)
#         done = terminated or truncated

#         Q[state,action] = Q[state,action] + learning_rate * (reward + discount_factor * np.max(Q[next_state]-Q[state,action]))

#         state = next_state
#     print("Q Table",Q)



