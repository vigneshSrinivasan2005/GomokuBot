import numpy as np
import pandas as pd
import random
import tensorflow as tf

class VigeAgent:
    def __init__(self, player, alpha, board_size):
        self.batch = pd.DataFrame(columns = ["state", "value"])
        self.alpha = alpha
        self.player = player
        self.board_size = board_size
        self.__to_int_vector1 = np.array([3 ** i for i in range(self.board_size)])
        self.__to_int_vector2 = np.array([3 ** (i * self.board_size) for i in range(self.board_size)])
        self.currPos = None

        
    #converts base 10 integer into base 3 integer into array (DON'T USE, THIS IS SLOW)
    def _toArray(self, input):
        array = []
        temp = input
        for i in range(self.board_size):
            row = []
            for j in range(self.board_size):
                row.append(temp % 3)
                temp = temp//3
            array.append(row)
        return array

    #converts array of game state into base 10 integer
    def _toInt(self, array):
        out = np.dot(np.dot(array, self.__to_int_vector1), self.__to_int_vector2)
        return out
    
    def _getLegalMoves(self, state):
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

    def _getStateValue(self, state):
        state = self._toArray(state)
        if(state == None):
            return 0
        temp = np.array(state).flatten()
        input = [temp]
        input = np.array(input)
        out = self.model(input, training = False)
        return out[0]
    
    def _getNextState(self, state, move):
        #print(str(move))
        return state + move

    def getMove(self, state, epsilon):
        legal_moves = self._getLegalMoves(state)
        best_move = None
        best_score = -1000000 if self.player == 1 else 1000000
        for move in legal_moves:
            next = self._getNextState(state, move)
            value = self._getStateValue(next) * (1 if self.player == 1 else -1)
            if(value >= best_score):
                best_move = move
                best_score = value
        if (random.random() < epsilon):
            #print("random")
            best_move = random.choice(legal_moves)
        self.currPos = self._getNextState(state, best_move)
        return best_move

    def update(self):
        x = np.array(self.batch["state"].to_list())
        new_x = []
        for value in x:
            new_x.append(np.array(self._toArray(value)).flatten())
        new_x = np.array(new_x)
        #print(x, " x")
        y = np.array(self.batch["value"].to_list())
        #print(y, " y")
        self.model.train_on_batch(new_x, y)
        #print("emptying batch")
        self.batch = pd.DataFrame(columns = ["state", "value"])


    def updateBatch(self, reward):
        #print(self.batch)
        #print("sad ", state, " ", next_state)
        y = reward + self.gamma * tf.get_static_value(self._getStateValue(self.currPos))[0]
        x = self.currPos
        if not (x in self.batch["state"].values):
            self.batch.loc[len(self.batch.index)] = [x, 0.0]
        self.batch.loc[self.batch["state"] == x, "value"] += (self.alpha * (y - self.batch.loc[self.batch["state"] == x, "value"].iloc[0]))
        #print(self.batch)
        #print("done w/", self.player)
    def save(self, name):
        self.model.save(name)