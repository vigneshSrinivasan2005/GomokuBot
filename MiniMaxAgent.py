from Agent import Agent
import pandas as pd
import random
class MiniMaxAgent(Agent):
    plict={}
    miniMax=pd.read_csv("modifiedMinMax.csv")
    def __init__(self,player,alpha,board_size):
        super().__init__(player,alpha,board_size)
        pass
    def getMove(self, state, epsilon):
        legal_moves = self._getLegalMoves(state)
        best_move = None
        best_score = -1000000 if self.player == 1 else 1000000
        for move in legal_moves:
            next = self._getNextState(state, move)
            value = self._getStateValue(next)
            if(self.player == 1):
                if(value >= best_score):
                    best_move = move
                    best_score = value
            else:
                if(value <= best_score):
                    best_move = move
                    best_score = value

            
        if (random.random() < epsilon):
            print("random")
            best_move = random.choice(legal_moves)
        nextState = self._getNextState(state, best_move)
        #arrayState = self._toArray(nextState)
        #print(arrayState[0])
        #print(arrayState[1])
        #print(arrayState[2])
        #print()
        #dicts cannot store duplicates
        self.plict[nextState]=best_score*(1 if self.player == 1 else -1)
        self._updateAgent(nextState)

        return best_move
    def savePlict(self,name="plict.csv"):
        df= pd.DataFrame(columns=["State","Value"])
        for key in self.plict:
            df.loc[len(df)]=[key,self.plict[key]]
        df.to_csv(name,index=False)
    def calculateMeanSquareError(self):
        error=0
        for key in self.plict:
            error+=(self.plict[key]-(self.miniMax[self.miniMax["State"]==key].values[0][1]/10))**2
        return error