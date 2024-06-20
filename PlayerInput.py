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
    model.add(layers.Input(shape = (10,)))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(32, activation = 'tanh'))
    model.add(layers.Dense(32, activation = 'tanh'))
    #model.add(layers.Dense(32, activation = 'relu'))
    #model.add(layers.Dense(32, activation = 'linear'))
    model.add(layers.Dense(1, activation = 'linear'))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.MeanSquaredError())
    def __toArray(self, input):
        array = []
        temp = input
        for i in range(self.board_size):
            row = []
            for j in range(self.board_size):
                row.append(temp % 3)
                temp = temp//3
            array.append(row)
        return array
    def __getStateValue(self, state):
        state = self.__toArray(state)
        if(state == None):
            return 0
        input = np.array(state).flatten()
        input= np.insert(input, 0, self.player-1)
        input = [input]
        input = np.array(input)
        out = self.model(input, training = False)
        return out[0]
    def updateBatch(self, reward):
        #print(self.batch)
        #print("sad ", state, " ", next_state)
        if(self.this_state_action == None):
            y = reward
            x = self.last_state_action

            if not (x in self.batch["state"].values):
                self.batch.loc[len(self.batch.index)] = [x, 0.0]

            #TODO is bugged
            #Takes the value of the state_action_pair x', in the batch, and updates it to be equal to x' + alpha *(y - x') 
            # where y is equal to the value estimate given by the current model estimate of the next state action pair and the reward
            self.batch.loc[self.batch["state"] == x, "value"] += self.alpha * (y - self.batch.loc[self.batch["state"] == x, "value"].iloc[0])


        elif(self.last_state_action != None):
            #print(state)
            y = reward + self.gamma * tf.get_static_value(self.__getStateValue(self.this_state_action))[0]
            #print(tf.get_static_value(y)[0], " y")
            x = self.last_state_action

            if not (x in self.batch["state"].values):
                self.batch.loc[len(self.batch.index)] = [x, 0.0]
            #TODO is bugged
            self.batch.loc[self.batch["state"] == x, "value"] += (self.alpha * (y - self.batch.loc[self.batch["state"] == x, "value"].iloc[0]))
        #print(self.batch)
        #print("done w/", self.player)
    def __getLegalMoves(self, state):
        cur_state = state
        legal_moves = []
        move = 1
        for i in range(self.board_size):
            for j in range(self.board_size):
                if(cur_state % 3 == 0):
                    legal_moves.append(move*self.player)
                cur_state = cur_state // 3
                move *= 3
        return legal_moves
    def __updateAgent(self, new_state_action):
        self.last_state_action = self.this_state_action
        self.this_state_action = new_state_action

    def __getNextState(self, state, move):
        #print(str(move))
        return state + move
    def __init__(self, player, alpha, board_size, input_model=None):
        super().__init__(player, alpha, 3)
        if input_model != None:
            self.model = input_model
        self.board_size=board_size
    def getMove(self, state, epsilon):
        legal_moves = self.__getLegalMoves(state)
        best_move = None
        best_score = -1000000
        for move in legal_moves:
            next = self.__getNextState(state, move)
            value = self.__getStateValue(next)
            if(value >= best_score):
                best_move = move
                best_score = value
        if (random.random() < epsilon):
            print("random")
            best_move = random.choice(legal_moves)
        self.__updateAgent(self.__getNextState(state, best_move))
        return best_move
    def update(self):
        x = np.array(self.batch["state"].to_list())
        new_x = [self.player-1]
        for value in x:
            new_x.append(np.array(self.__toArray(value)).flatten())
        new_x = np.array(new_x)
        #print(x, " x")
        y = np.array(self.batch["value"].to_list())
        #print(y, " y")
        self.model.train_on_batch(new_x,y)
        #print("emptying batch")
        self.batch = pd.DataFrame(columns = ["state", "value"])