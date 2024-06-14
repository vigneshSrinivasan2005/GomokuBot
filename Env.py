# Agent has the following
#this_state_value->2d array
#last_state_value->2d array
#getMove(State)
#update(Batch)
import pandas as pd
import random
from Gomoku import Gomoku
class Env:
    def __init__(self, board_size, win_con, agent1, agent2, batchSize):
        self.agent1 = agent1
        self.agent2 = agent2
        self.batchSize = batchSize
        self.board_size = board_size
        self.win_con = win_con
    def playBatch(self):
        for i in range(self.batchSize):
            self.__playGame()
        self.agent1.update(self.batch1)
        self.agent2.update(self.batch2)

    def __playGame(self):   
        game = Gomoku(self.board_size, self.win_con)
        #game.game()
        winner = game.getWinner()
        self.agent1.last_state_action = None
        self.agent1.this_state_action = None
        self.agent2.last_state_action = None
        self.agent2.this_state_action = None
        r1 = 0
        r2 = 0

        while(winner == -1):
            best_move=self.agent1.getMove(game.getState())

            game.playMove(best_move)
            
            #update p1 with previous state action and reward
            self.agent1.updateBatch(r1)

            #get reward r1
            r1 = game.getReward()

            #if p1 wins update agent for p2 with -r1 and update p1 with r1 and end
            #p1 will be updated with it's state action pair leading to an end state, as it's updates have always been one step behind.
            winner = game.getWinner()
            if(winner == 1 or winner == 0):
                break

            best_move=self.agent2.getMove(game.getState())

            game.playMove(best_move)
            
            #update p1 with previous state action and reward
            self.agent2.updateBatch(r2)

            #get reward r1
            r2 = game.getReward()

            #if p1 wins update agent for p2 with -r1 and update p1 with r1 and end
            #p1 will be updated with it's state action pair leading to an end state, as it's updates have always been one step behind.
            winner = game.getWinner()
            if(winner == 1 or winner == 0):
                break