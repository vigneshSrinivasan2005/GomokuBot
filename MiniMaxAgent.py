from Agent import Agent
import pandas as pd
import numpy as np
import random
class MiniMaxAgent(Agent):
    minimax= pd.read_csv("modifiedMiniMax.csv")
    def __init__(self,player,alpha,board_size):
        super().__init__(player,alpha,board_size)
        pass

    def getMove(self, state, epsilon):
        legal_moves = self._getLegalMoves(state)
        best_move = None
        best_score = -10000
        for move in legal_moves:
            next = self._getNextState(state, move)
            value = (-1 if self.player == 2 else 1) * self._getStateValue(next)
            #print(next," ", value)
            if(value >= best_score):
                best_move = move
                best_score = value
        if (random.random() < epsilon):
            #print("random")
            best_move = random.choice(legal_moves)
        
        self._updateAgent(state, best_move)
        
        return best_move
    
    def _updateAgent(self, state, move):
        self.last_state_action = state
        self.this_state_action = state + move
    def evaluate(self):
        x = np.array(self.batch["State"].to_list())
        new_x = []
        for value in x:
            new_x.append(np.array(self._toArray(value)).flatten())
        new_x = np.array(new_x)
        #print(x, " x")
        y = np.array(self.batch["value"].to_list())
        self.model.evaluate(new_x,y)
    