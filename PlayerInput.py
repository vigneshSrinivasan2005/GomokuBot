from Agent import Agent
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import tensorflow as tf
import random
class PlayerInput(Agent):
    gamma = 1
    model = None
    model = keras.Sequential()
    model.add(layers.Input(shape = ((3*3) +1,)))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(32, activation = 'tanh'))
    #model.add(layers.Dense(32, activation = 'relu'))
    #model.add(layers.Dense(32, activation = 'linear'))
    model.add(layers.Dense(1, activation = 'linear'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError())

    def __init__(self, player, alpha, input_model=None):
        super().__init__(player, alpha, 3)
        if input_model != None:
            self.model = input_model
    
    def _getStateValue(self, state):
        state = self._toArray(state)
        if(state == None):
            return 0
        input = np.array(state).flatten()
        input= np.insert(input, 0, self.player)
        input = [input]
        input = np.array(input)
        out = self.model(input, training = False)
        return out[0]
    

    
    def update(self):
        x = np.array(self.batch["state"].to_list())
        new_x = []
        for value in x:
            temp = np.array(self._toArray(value)).flatten()
            temp = np.insert(temp, 0, self.player)
            new_x.append(temp)
        new_x = np.array(new_x)
        #print(x, " x")
        y = np.array(self.batch["value"].to_list())
        #print(y, " y")
        self.model.train_on_batch(new_x,y)
        #print("emptying batch")
        self.batch = pd.DataFrame(columns = ["state", "value"])
        self.last_state_action = None
        self.this_state_action = None
        self.opp_last_state_action = None
        self.opp_this_state_action = None


        def _updateBatchHelper(self, last_state_action, this_state_action, reward ):
            #print(self.batch)
            #print("sad ", state, " ", next_state)
            if(this_state_action == None and last_state_action != None):
                y = reward
                x = last_state_action

                if not (x in self.batch["state"].values):
                    self.batch.loc[len(self.batch.index)] = [x, 0.0]

                #TODO is bugged
                #Takes the value of the state_action_pair x', in the batch, and updates it to be equal to x' + alpha *(y - x') 
                # where y is equal to the value estimate given by the current model estimate of the next state action pair and the reward
                self.batch.loc[self.batch["state"] == x, "value"] += self.alpha * (y - self.batch.loc[self.batch["state"] == x, "value"].iloc[0])


            elif(last_state_action != None):
                #print(state)
                y = reward + self.gamma * tf.get_static_value(self._getStateValue(self.this_state_action))[0]
                #print(tf.get_static_value(y)[0], " y")
                x = last_state_action

                if not (x in self.batch["state"].values):
                    self.batch.loc[len(self.batch.index)] = [x, 0.0]
                #TODO is bugged
                self.batch.loc[self.batch["state"] == x, "value"] += (self.alpha * (y - self.batch.loc[self.batch["state"] == x, "value"].iloc[0]))
            #print(self.batch)
            #print("done w/", self.player)