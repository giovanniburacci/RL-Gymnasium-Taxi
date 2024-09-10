import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import sleep

environment = gym.make("Taxi-v3", render_mode="ansi")
environment.reset()
qtable = np.zeros(
    (environment.observation_space.n,
     environment.action_space.n)
)


episodes = 10000        # Total number of episodes
learning_rate = 0.3     # Learning rate
gamma = 0.9            # Discount factor
epsilon = 1
epsilon_decay = 0.995
epsilon_min = 0.01

# Training
total_reward = 0
rewards = []
for _ in range(episodes):
    state = environment.reset()[0]
    done = False
    # By default, we consider our outcome to be a failure

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:

        rnd = np.random.random()

        # prefer exploration first, exploitation later
        # chooses random action with probability epsilon
        if rnd < epsilon:
            action = environment.action_space.sample()
        # chooses best action with probability 1-epsilon
        else:
            action = np.argmax(qtable[state])

        # Implement this action and move the agent in the desired direction
        new_state, reward, terminated, truncated, _ = environment.step(action)
        total_reward += reward
        done = terminated or truncated
        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
                                learning_rate * (
                                        reward + gamma * np.max(qtable[new_state])
                                                 - qtable[state, action])
        # update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update our current state
        state = new_state
    rewards.append(total_reward)
    total_reward = 0

# smoothening rewards
smooth_rewards = pd.DataFrame(rewards).rolling(20).mean()

# Plotting the rewards over episodes
plt.plot(smooth_rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Tabular Q-Learning Training Performance Over Episodes')
plt.savefig('learning_plot.png')
plt.show()

window_size = 100
avg_rewards = [np.mean(rewards[i:i+window_size]) for i in range(0, len(rewards), window_size)]
reward_threshold = 6
for i, avg_reward in enumerate(avg_rewards):
    # set a threshold for arbitrarily measuring 'good performance'
    if avg_reward > reward_threshold:
        print(f"Convergence achieved at episode {i * window_size}")
        break


episodes = 10000
nb_success = 0


for _ in range(episodes):
    state = environment.reset()[0]
    done = False

    # Until the agent gets stuck or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        # argmax returns the key of the highest value in the array (action)
        action = np.argmax(qtable[state])

        # Implement this action and move the agent in the desired direction
        new_state, reward, terminated, truncated, _ = environment.step(action)
        # print(environment.render())
        done = terminated or truncated
        # Update our current state
        state = new_state
        if terminated:
            nb_success += 1

# let's check our success rate!
print(f"Success rate = {nb_success/episodes*100}%")
environment.close()