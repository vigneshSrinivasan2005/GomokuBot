from MiniMaxAgent import MiniMaxAgent
import keras
from keras import layers

class DefaultMiniMaxAgent(MiniMaxAgent):
    gamma = 1
    model = None
    model = keras.Sequential()
    model.add(layers.Input(shape = (3 * 3,)))
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(32, activation = 'relu'))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(1, activation = 'linear'))
    model.compile(optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredError())
    def __init__(self, player, alpha, input_model=None):
        super().__init__(player, alpha, 3)
        if input_model != None:
            self.model = input_model
    