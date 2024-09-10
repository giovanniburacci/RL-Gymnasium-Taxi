import gymnasium as gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory
from nn_model import DQN
import pandas as pd

# Hyperparameters
env = gym.make("Taxi-v3", render_mode=None)
state_size = env.observation_space.n
action_size = env.action_space.n

learning_rate = 0.0005
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.998
epsilon_min = 0.01
batch_size = 64
memory_capacity = 10000
num_episodes = 10000

# Initialize DQN and memory
model = DQN(action_size)
memory = ReplayMemory(memory_capacity)

# Build model by passing some input data (state_one_hot) through it
dummy_state = tf.one_hot([0], state_size)  # Use a dummy state to initialize the model
model(dummy_state)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_function = tf.keras.losses.MeanSquaredError()

# List to store rewards for plotting
rewards_history = []

def choose_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    state_one_hot = tf.one_hot(state, state_size)  # One-hot encode the state
    q_values = model(np.array([state_one_hot], dtype=np.float32))
    return np.argmax(q_values[0])  # Choose action with the highest Q-value

def train_step():
    if len(memory) < batch_size:
        return

    batch = memory.sample(batch_size)
    states, actions, rewards, next_states, terminated = zip(*batch)

    # Convert to NumPy arrays
    states = np.array(states, dtype=np.int32)
    next_states = np.array(next_states, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    terminated = np.array(terminated, dtype=np.float32)

    # One-hot encode the states and next states
    states_one_hot = tf.one_hot(states, state_size)
    next_states_one_hot = tf.one_hot(next_states, state_size)

    # Target q values are computed for the next state
    target_qs = model(next_states_one_hot)

    # selecting best action according to target q values
    max_next_qs = np.amax(target_qs, axis=1)

    # target q value computed using the Bellman equation
    # (1 - terminated) is used to ignore future rewards
    # from terminal (successfully) states
    target_values = rewards + gamma * max_next_qs * (1 - terminated)

    # GradientTape is used to record operations on the model's weights
    with tf.GradientTape() as tape:

        # computing q values for all actions on current state
        qs = model(states_one_hot)

        # one hot encoding of actions
        action_masks = tf.one_hot(actions, action_size)

        # multiply q values of each action times the actions that were really executed (one hot encoding)
        # so that we only consider actions taken, and we sum the values and reduce it on the columns
        masked_qs = tf.reduce_sum(qs * action_masks, axis=1)

        # compute loss function between the target values, computed through the Bellman equation
        # and the q values returned by our neural network
        loss = loss_function(target_values, masked_qs)

    # compute gradients of loss function
    grads = tape.gradient(loss, model.trainable_variables)
    # update model weights with computer gradients (BACKPROPAGATION STEP)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Training loop
for episode in range(num_episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False

    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)

        # Store transition in memory
        memory.push((state, action, reward, next_state, terminated))

        state = next_state
        total_reward += reward
        train_step()

        # End episode if terminated or truncated
        done = terminated or truncated

        if done:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            rewards_history.append(total_reward)  # Record the total reward after each episode
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

# Plotting the rewards over episodes
smooth_rewards = pd.DataFrame(rewards_history).rolling(20).mean()
plt.plot(smooth_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQL Training Performance Over Episodes')
plt.savefig('learning_plot_less-decay.png')
plt.show()

nb_success = 0
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()[0]
    total_reward = 0
    done = False
    steps = 0

    while not done:
        env.render()
        state_one_hot = tf.one_hot(state, state_size)
        q_values = model(np.array([state_one_hot], dtype=np.float32))
        action = np.argmax(q_values[0])
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        if terminated:
            nb_success += 1
    print(f"Episode {episode + 1}: Completed in {steps} steps with total reward: {total_reward}")

print(f"Success rate = {nb_success/num_episodes*100}%")
env.close()
