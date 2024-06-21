import math
from Agent import Agent
from tensorflow import keras
from keras import layers
import pandas as pd
class Agent3x3UCB(Agent):
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
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError())

    def __init__(self, player, alpha, c, input_model=None):
        super().__init__(player, alpha, 3)
        self.c = c
        self.time_steps = 1
        self.N = pd.DataFrame(columns = ["state", "times"])
        if input_model != None:
            self.model = input_model
            
        

    def getMove(self, state, epsilon = 0):
        legal_moves = self._getLegalMoves(state)
        best_move = None
        best_score = -1000000
        for move in legal_moves:
            next = self._getNextState(state, move) 
            value = self._getStateValue(next) + self.c * math.sqrt(math.log(self.time_steps)/self._getN(next))
            if(value >= best_score):
                best_move = move
                best_score = value

        self._updateAgent(self._getNextState(state, best_move))
        
        return best_move
    def _updateAgent(self, new_state_action):
        super()._updateAgent(new_state_action)
        self.time_steps += 1
        if not (new_state_action in self.N["state"].values):
                self.N.loc[len(self.N.index)] = [new_state_action, 1]
        self.N.loc[self.N["state"] == new_state_action, "times"] += 1

    def _getN(self, new_state_action):
         if not (new_state_action in self.N["state"].values):
            self.N.loc[len(self.N.index)] = [new_state_action, 1]
            return 1
         #print(self.N.loc[self.N["state"] == new_state_action, "times"])
         return self.N.loc[self.N["state"] == new_state_action, "times"].iloc[0]
         
    