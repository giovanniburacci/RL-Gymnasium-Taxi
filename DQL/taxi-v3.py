import gymnasium as gym
import numpy as np
import tensorflow as tf
from replay_memory import ReplayMemory
from nn_model import DQN

# Hyperparameters
env = gym.make("Taxi-v3", render_mode=None)
state_size = env.observation_space.n
action_size = env.action_space.n

learning_rate = 0.001
gamma = 0.99  # discount factor
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 32
memory_capacity = 10000
num_episodes = 4000

# Initialize DQN and memory
model = DQN(action_size)
memory = ReplayMemory(memory_capacity)

# Build model by passing some input data (state_one_hot) through it
dummy_state = tf.one_hot([0], state_size)  # Use a dummy state to initialize the model
model(dummy_state)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_function = tf.keras.losses.MeanSquaredError()

# before computing the q values, we need to one hot the states (transformed into a binary vector with all 0s, except for
# one 1 which corresponds to the current state)
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
    states, actions, rewards, next_states, dones = zip(*batch)

    # Convert to NumPy arrays
    states = np.array(states, dtype=np.int32)
    next_states = np.array(next_states, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)

    # One-hot encode the states and next states
    states_one_hot = tf.one_hot(states, state_size)
    next_states_one_hot = tf.one_hot(next_states, state_size)

    # Compute target Q-values using the same model

    # output of the dqn, list of target q values for each action given the input state
    target_qs = model(next_states_one_hot)

    # maximum target q value
    max_next_qs = np.amax(target_qs, axis=1)

    # Bellman equation
    target_values = rewards + gamma * max_next_qs * (1 - dones)

    with tf.GradientTape() as tape:
        qs = model(states_one_hot)
        action_masks = tf.one_hot(actions, action_size)
        masked_qs = tf.reduce_sum(qs * action_masks, axis=1)
        loss = loss_function(target_values, masked_qs)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

# Training loop
for episode in range(num_episodes):
    state = env.reset()[0]
    total_reward = 0

    while True:
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.push((state, action, reward, next_state, done))

        state = next_state

        total_reward += reward
        train_step()

        if done:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")
            break

num_episodes = 100
nb_success = 0
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
        done = terminated or truncated
        total_reward += reward
        if reward == 20:
            nb_success += 1
        steps += 1

    print(f"Episode {episode + 1}: Completed in {steps} steps with total reward: {total_reward}")

print(f"Success rate = {nb_success/num_episodes*100}%")

env.close()
