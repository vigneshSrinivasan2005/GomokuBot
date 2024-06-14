import random
import numpy as np
import pandas as pd
import tensorflow as tf

class Agent:
    
    def __init__(self, player, alpha):
        self.last_state_value = None
        self.this_state_value = None
        self.batch = pd.DataFrame(columns = ["state", "value"])
        self.alpha = alpha
        self.player = player
        

    def __updateAgent(self, new_state_value):
        self.last_state_value = self.this_state_value
        self.this_state_value = new_state_value

    def __getLegalMoves(self, state):
        legal_moves = []
        for i in range(len(state)):
            for j in range(len(state[i])):
                if(state[i][j] == 0):
                    legal_moves.append([i, j])
        return legal_moves

    def __getStateValue(self, state):
        if(state == None):
            return 0
        temp = np.array(state).flatten()
        input = [temp]
        input = np.array(input)
        out = self.model(input, training = False)
        return out[0]
    
    def __getNextState(self, state, move):
        #print(str(move))
        state[move[0]][move[1]] = self.player
        return state

    def getMove(self, state, epsilon):
        legal_moves = self.__getLegalMoves(state)
        best_move = None
        best_score = -1000
        for move in legal_moves:
            next = self.__getNextState(state, move)
            value = self.__getStateValue(next)
            if(value >= best_score):
                best_move = move
                best_score = value
        if (random.random() < epsilon):
            best_move = random.choice(legal_moves)

        self.__updateAgent(self.__getNextState(state, best_move))
        return best_move

    def update(self):
        x = np.array(self.batch["state"].to_list())
        #print(x, " x")
        y = np.array(self.batch["value"].to_list())
        #print(y, " y")
        self.model.train_on_batch(x, y)

    def updateBatch(self, reward):
        #print("sad ", state, " ", next_state)
        if(self.this_state_value == None):
            y = reward
            x = np.array(self.last_state_value).flatten()
            if not (self.batch['state'].apply(lambda input: np.array_equal(input, x)).any()):
                self.batch.loc[len(self.batch.index)] = [x, 0]
            con1 = self.batch["state"].apply(lambda input: np.array_equal(input, x))
            self.batch.loc[con1, 'value'] = self.batch.loc[con1, 'value'] + self.alpha * (y - self.batch.loc[con1, 'value'])

        elif(self.last_state_value != None):
            #print(state)
            y = reward + self.gamma * self.__getStateValue(self.this_state_value)
            #print(tf.get_static_value(y)[0], " y")
            x = np.array(self.last_state_value).flatten()
            if not (self.batch['state'].apply(lambda input: np.array_equal(input, x)).any()):
                self.batch.loc[len(self.batch.index)] = [x, 0]
            con1 = self.batch["state"].apply(lambda input: np.array_equal(input, x))
            self.batch.loc[con1, 'value'] = self.batch.loc[con1, 'value'] + self.alpha * (tf.get_static_value(y)[0] - self.batch.loc[con1, 'value'])
    def save(self, name):
        self.model.save(name)