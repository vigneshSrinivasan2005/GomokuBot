import pandas as pd
import numpy as np
from tensorflow import keras
from keras import layers
from Agent import Agent
#inputs to train on batch
#2dNPArray of states
#1dNPArray of values
class MinimaxModel(Agent):
    def __init__(self, board_size, file, player, model = None):
        super().__init__(player, 0, board_size)
        self.board_size = board_size
        # data is a numpy array of numpy arrays
        
        self.data= pd.read_csv(file)

        
        if(model != None):
            self.model = model
        else:
            model = keras.Sequential()
            model.add(layers.Input(shape = (board_size * board_size,)))
            model.add(layers.Dense(32, activation = 'relu'))
            model.add(layers.Dense(32, activation = 'relu'))
            model.add(layers.Dense(32, activation = 'tanh'))
            model.add(layers.Dense(32, activation = 'relu'))
            model.add(layers.Dense(32, activation = 'tanh'))
            model.add(layers.Dense(32, activation = 'relu'))
            model.add(layers.Dense(1, activation = 'linear'))
            model.compile(optimizer=keras.optimizers.Adam(),
                            loss=keras.losses.MeanSquaredError())
            self.model=model
            self.train()
    def train(self):
        x = np.array(self.data["State"].to_list())
        new_x = []
        for value in x:
            new_x.append(np.array(self._toArray(value)).flatten())
        new_x = np.array(new_x)
        #print(x, " x")
        y = np.array(self.data["Value"].to_list())
        #print(y, " y")
        self.model.fit(new_x, y, epochs = 1000,batch_size = 50)
    def save(self, name):
        self.model.save(name)