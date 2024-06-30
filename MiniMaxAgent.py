from Agent import Agent
import pandas as pd
import random
class MiniMaxAgent(Agent):

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

    