# Agent has the following
#this_state_value->2d array
#last_state_value->2d array
#getMove(State)
#update(Batch)
from Gomoku import Gomoku

class Env:
    #inputs are board size, win condition, agent1, agent2, and batch size
    def __init__(self, board_size, win_con, agent1, agent2, batchSize):
        self.agent1 = agent1
        self.agent2 = agent2
        self.batchSize = batchSize
        self.board_size = board_size
        self.win_con = win_con
    
    #plays batch of games
    def playBatch(self):
        for i in range(self.batchSize):
            self.__playGame()
        self.agent1.update()
        self.agent2.update()
    
    #plays single game
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
            #player 1
            #gets move
            best_move=self.agent1.getMove(game.getState())
            game.playMove(best_move)
            #updates batch in agent
            self.agent1.updateBatch(r1)
            #gets reward
            r1 = game.getReward()
            #checks if game is over
            winner = game.getWinner()
            if(winner == 1 or winner == 0):
                break
            #player 2
            #gets move
            best_move=self.agent2.getMove(game.getState())
            game.playMove(best_move)
            #updates batch in agent
            self.agent2.updateBatch(r2)
            #gets reward
            r2 = game.getReward()
            #checks if game is over
            winner = game.getWinner()
            if(winner == 1 or winner == 0):
                break