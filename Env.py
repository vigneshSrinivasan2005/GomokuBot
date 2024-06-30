# Agent has the following
#this_state_value->2d array
#last_state_value->2d array
#getMove(State)
#update(Batch)
from Gomoku import Gomoku

class Env:
    #inputs are board size, win condition, agent1, agent2, and batch size
    def __init__(self, board_size, win_con, agent1, agent2, batch_size,epsilon=0.2,agent1_is_training=True,agent2_is_training=True):
        self.agent1 = agent1
        self.agent2 = agent2
        self.batchSize = batch_size
        self.board_size = board_size
        self.win_con = win_con
        self.epsilon = epsilon
        self.agent1_is_training = agent1_is_training
        self.agent2_is_training = agent2_is_training
    #plays batch of games
    def playBatch(self):
        for i in range(self.batchSize):
            #print("game: ", i)
            self.__playGame()
        #Add ONLY if batching
        #if(self.agent1_is_training):
            #self.agent1.update()
        #if(self.agent2_is_training):
            #self.agent2.update()
    
    #plays single game
    def __playGame(self):   
        game = Gomoku(self.board_size, self.win_con)
        #game.game()
        winner = game.getWinner()

        r1 = 0
        r2 = 0

        while(winner == -1):
            #player 1
            #gets move
            best_move=self.agent1.getMove(game.getState(), self.epsilon)
            game.playMove(best_move)
            #updates batch in agent
            if(self.agent1_is_training):
                self.agent1.updateSingle(r1)
            #gets reward
            r1 = game.getReward()
            #checks if game is over
            winner = game.getWinner()
            if(winner == 1 or winner == 0):
                if(self.agent1_is_training):
                    self.agent1.updateSingle(r1)
                if(self.agent2_is_training):
                    self.agent2.updateSingle(r2)
                break
            #player 2
            #gets move
            best_move=self.agent2.getMove(game.getState(),self.epsilon)
            game.playMove(best_move)
            #updates batch in agent
            if(self.agent2_is_training):
                self.agent2.updateSingle(r2)
            #gets reward
            r2 = game.getReward()
            #checks if game is over
            winner = game.getWinner()
            if(winner == 2 or winner == 0):
                if(self.agent1_is_training):
                    self.agent1.updateSingle(r1)
                if(self.agent2_is_training):
                    self.agent2.updateSingle(r2)
                break


