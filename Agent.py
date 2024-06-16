import random
import numpy as np
import pandas as pd
import tensorflow as tf

class Agent:
    
    def __init__(self, player, alpha, board_size):
        self.last_state_action = None
        self.this_state_action = None
        self.batch = pd.DataFrame(columns = ["state", "value"])
        self.alpha = alpha
        self.player = player
        self.board_size = board_size
        
    #converts base 10 integer into base 3 integer into array (DON'T USE, THIS IS SLOW)
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

    #converts array of game state into base 10 integer
    def __toInt(self, array):
        out = np.dot(np.dot(array, self.__to_int_vector1), self.__to_int_vector2)
        return out
    
    def __updateAgent(self, new_state_action):
        self.last_state_action = self.this_state_action
        self.this_state_action = new_state_action

    def __getLegalMoves(self, state):
        cur_state = state
        legal_moves = []
        move = 1
        for i in range(self.board_size):
            for j in range(self.board_size):
                if(cur_state % 3 == 0):
                    legal_moves.append(move)
                move *= 3
        return legal_moves

    def __getStateValue(self, state):
        state = self.__toArray(state)
        if(state == None):
            return 0
        temp = np.array(state).flatten()
        input = [temp]
        input = np.array(input)
        out = self.model(input, training = False)
        return out[0]
    
    def __getNextState(self, state, move):
        #print(str(move))
        return state + move

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
            best_move = random.choice(legal_moves)

        self.__updateAgent(self.__getNextState(state, best_move))
        return best_move

    def update(self):
        x = np.array(self.batch["state"].to_list())
        new_x = []
        for value in x:
            new_x.append(self.__toArray(value))
        new_x = np.array(new_x)
        #print(x, " x")
        y = np.array(self.batch["value"].to_list())
        #print(y, " y")
        self.model.train_on_batch(x, y)

    def updateBatch(self, reward):
        #print("sad ", state, " ", next_state)
        if(self.this_state_action == None):
            y = reward
            x = self.last_state_action

            if not (x in self.batch["state"].values):
                self.batch.loc[len(self.batch.index)] = [x, 0]


            #Takes the value of the state_action_pair x', in the batch, and updates it to be equal to x' + alpha *(y - x') 
            # where y is equal to the value estimate given by the current model estimate of the next state action pair and the reward
            self.batch.loc[self.batch["state"] == x, "value"].iloc[0] += self.alpha * (y - self.batch.loc[self.batch["state"] == x, "value"].iloc[0])


        elif(self.last_state_action != None):
            #print(state)
            y = reward + self.gamma * self.__getStateValue(self.this_state_action)
            #print(tf.get_static_value(y)[0], " y")
            x = self.last_state_action

            if not (x in self.batch["state"].values):
                self.batch.loc[len(self.batch.index)] = [x, 0]

            self.batch.loc[self.batch["state"] == x, "value"].iloc[0] += self.alpha * (tf.get_static_value(y)[0] - self.batch.loc[self.batch["state"] == x, "value"].iloc[0])

    def save(self, name):
        self.model.save(name)