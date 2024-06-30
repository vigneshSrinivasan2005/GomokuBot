from Agent import Agent
import pandas as pd
import numpy as np
import random
class MiniMaxAgent(Agent):
    minimax= pd.read_csv("modifiedMinMax.csv")
    batch = pd.DataFrame(columns = ["state", "value"])

    def __init__(self,player,alpha,board_size):
        self.last_state_action = None
        self.this_state_action = None
        self.alpha = alpha
        self.player = player
        self.board_size = board_size
        self.gamma = 0.99
        self.__to_int_vector1 = np.array([3 ** i for i in range(self.board_size)])
        self.__to_int_vector2 = np.array([3 ** (i * self.board_size) for i in range(self.board_size)])
        pass

    def getMove(self, state, epsilon):
        legal_moves = self._getLegalMoves(state)
        best_move = None
        best_score = -10000
        for move in legal_moves:
            next = self._getNextState(state, move)
            value = (-1 if self.player == 2 else 1) * self._getStateValue(next)
            #print(next," ", value)
            #print(next," ", self._getStateValue(next))
            if(value >= best_score):
                best_move = move
                best_score = value
        if (random.random() < epsilon):
            #print("random")
            best_move = random.choice(legal_moves)
        
        self._updateAgent(state, best_move)
        #print("chose:", self._getNextState(state, best_move))
        
        return best_move
    
    def _updateAgent(self, state, move):
        self.last_state_action = state
        self.this_state_action = state + move
    def evaluate(self):
        x = np.array(self.minimax["State"].to_list())
        new_x = []
        for value in x:
            new_x.append(np.array(self._toArray(value)).flatten())
        new_x = np.array(new_x)
        #print(x, " x")
        y = np.array(self.minimax["Value"].to_list())
        self.model.evaluate(new_x,y)
    