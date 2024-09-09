import tensorflow as tf
from tensorflow.keras import layers

class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        # rectifier linear unit as activation function for hidden layers, since we are dealing with a regression problem
        self.fc1 = layers.Dense(24, activation='relu')
        self.fc2 = layers.Dense(24, activation='relu')
        # linear activation function
        self.fc3 = layers.Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)