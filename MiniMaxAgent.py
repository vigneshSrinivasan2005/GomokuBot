from Agent import Agent
import pandas as pd
import numpy as np
import tensorflow as tf
import random
class MiniMaxAgent(Agent):
    minimax= pd.read_csv("modifiedMinMax.csv")
    batch = pd.DataFrame(columns = ["state", "value", "count"])
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
        x = np.array(self.minimax["State"].to_list())
        new_x = []
        for value in x:
            new_x.append(np.array(self._toArray(value)).flatten())
        new_x = np.array(new_x)
        #print(x, " x")
        y = np.array(self.minimax["Value"].to_list())
        self.model.evaluate(new_x,y)

    def updateBatch(self, reward):
        print(MiniMaxAgent.batch)
        #print("sad ", state, " ", next_state)
        if self.last_state_action != None:
            last_state_prediction = tf.get_static_value(self._getStateValue(self.last_state_action))[0]
            this_state_prediction = tf.get_static_value(self._getStateValue(self.this_state_action))[0]
            
            y= last_state_prediction+self.alpha * (reward + self.gamma  * this_state_prediction-last_state_prediction) 

            x = self.last_state_action
            if not (x in MiniMaxAgent.batch["state"].values):
                MiniMaxAgent.batch.loc[len(MiniMaxAgent.batch.index)] = [x, y,1]
            else:
                MiniMaxAgent.batch.loc[MiniMaxAgent.batch["state"] == x, "value"] += y
                MiniMaxAgent.batch.loc[MiniMaxAgent.batch["state"] == x, "count"] += 1
        #is a terminal state so value is just the reward
        if(self.this_state_action == None):
            y = reward
            x = self.last_state_action

            if not (x in MiniMaxAgent.batch["state"].values):
                MiniMaxAgent.batch.loc[len(MiniMaxAgent.batch.index)] = [x, y,1]
            else:
                #TODO is bugged
                #Takes the value of the state_action_pair x', in the batch, and updates it to be equal to x' + alpha *(y - x') 
                # where y is equal to the value estimate given by the current model estimate of the next state action pair and the reward
                MiniMaxAgent.batch.loc[MiniMaxAgent.batch["state"] == x, "value"] += y
                MiniMaxAgent.batch.loc[MiniMaxAgent.batch["state"] == x, "count"] += 1
    def avgBatch(self):
        new_batch = pd.DataFrame(columns = ["state", "value"])
        for index, row in MiniMaxAgent.batch.iterrows():
            new_batch.loc[index, "state"] = row["state"]
            new_batch.loc[index, "value"] = row["value"]/row["count"]
        print(new_batch)
        MiniMaxAgent.batch = new_batch
    def update(self):
        if(MiniMaxAgent.batch.empty):
            return
        self.avgBatch()
        print(MiniMaxAgent.batch)
        x = np.array(MiniMaxAgent.batch["state"].to_list())
        new_x = []
        for value in x:
            new_x.append(np.array(self._toArray(value)).flatten())
        new_x = np.array(new_x)
        #print(x, " x")
        y = np.array(MiniMaxAgent.batch["value"].to_list())
        #print(y, " y")
        self.model.train_on_batch(new_x, y)
        #print("emptying batch")
        MiniMaxAgent.batch = pd.DataFrame(columns = ["state", "value","count"])
