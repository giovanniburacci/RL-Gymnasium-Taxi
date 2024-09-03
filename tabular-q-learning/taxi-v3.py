import gymnasium as gym
import numpy as np
import pygame


environment = gym.make("Taxi-v3", render_mode="ansi")
environment.reset()
environment.render()
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

episodes = 10000        # Total number of episodes
alpha = 0.5            # Learning rate
gamma = 0.9            # Discount factor
epsilon = 1
epsilon_decay = 0.001

# Training

for _ in range(episodes):
    state = environment.reset()[0]
    done = False

    # By default, we consider our outcome to be a failure

    # Until the agent gets stuck in a hole or reaches the goal, keep training it
    while not done:

        rnd = np.random.random()
        # Choose the action with the highest value in the current state

        if rnd < epsilon:
            action = environment.action_space.sample()
        else:
            action = np.argmax(qtable[state])

        # if np.max(qtable[state]) > 0:
        #     action = np.argmax(qtable[state])
        #
        # If there's no best action (only zeros), take a random one
        # else:
        #     action = environment.action_space.sample()

        # Implement this action and move the agent in the desired direction
        new_state, reward, terminated, truncated, _ = environment.step(action)

        done = terminated or truncated
        # Update Q(s,a)
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        # update epsilon
        epsilon = max(epsilon - epsilon_decay, 0)

        # Update our current state
        state = new_state


episodes = 100
nb_success = 0

# Evaluation
for _ in range(100):
    state = environment.reset()[0]
    done = False

    # Until the agent gets stuck or reaches the goal, keep training it
    while not done:
        # Choose the action with the highest value in the current state
        # if np.max(qtable[state]) > 0:
        #     action = np.argmax(qtable[state])
        #
        # # If there's no best action (only zeros), take a random one
        # else:
        #     action = environment.action_space.sample()

        action = np.argmax(qtable[state])

        # Implement this action and move the agent in the desired direction
        new_state, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        # Update our current state
        state = new_state
        if reward == 20:
            print(environment.render())
            nb_success += 1

# Let's check our success rate!
print(f"Success rate = {nb_success/episodes*100}%")