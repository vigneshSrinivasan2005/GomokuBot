import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import random
from Gomoku import Gomoku

SIZE = 3
ALPHA = 0.1

class Agent:
  gamma = 1
  model = None
  model = keras.Sequential()
  model.add(layers.Input(shape = (SIZE * SIZE,)))
  model.add(layers.Dense(32, activation = 'tanh'))
  model.add(layers.Dense(32, activation = 'tanh'))
  model.add(layers.Dense(32, activation = 'tanh'))
  model.add(layers.Dense(32, activation = 'tanh'))
  model.add(layers.Dense(32, activation = 'tanh'))
  model.add(layers.Dense(1, activation = 'linear'))
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
  loss=keras.losses.MeanSquaredError())


  def __init__(self):
    self.last_state_action = None
    self.this_state_action = None

  #def __init__(self, link, size):
  #  self.model = keras.models.load_model(link)
  #  SIZE = size

  def updateState(self, state):
    self.last_state_action = self.this_state_action
    self.this_state_action = state

  def getStateValue(self, state):
    if(state == None):
      return 0
    temp = np.array(state).flatten()
    input = [temp]
    input = np.array(input)
    out = self.model(input, training = False)
    return out[0]



  def updateAgent(self, state, next_state, reward):
    if(next_state == None):
      #print(state)
      target = reward
      y = [target]
      y = np.array(y)
      input = np.array(state).flatten()
      x = [input]
      x = np.array(x)
      self.model.train_on_batch(x, y)
    elif(state != None):
      #print(state)
      target = reward + self.gamma * self.getStateValue(next_state)
      y = [target]
      y = np.array(y)
      input = np.array(state).flatten()
      x = [input]
      x = np.array(x)
      self.model.train_on_batch(x, y)





  #updates agent given a dataframe of all the updates made this batch.
  def updateAgentBatch(self, batch):
    x = np.array(batch["state"].to_list())
    #print(x, " x")
    y = np.array(batch["value"].to_list())
    #print(y, " y")
    self.model.train_on_batch(x, y)








  #given a batch, updates the batch with the next state and rewards
  def updateBatch(self, state, next_state, reward, batch):
    #print("sad ", state, " ", next_state)
    if(next_state == None):
      y = reward
      x = np.array(state).flatten()
      if not (batch['state'].apply(lambda input: np.array_equal(input, x)).any()):
        batch.loc[len(batch.index)] = [x, 0]


      con1 = batch["state"].apply(lambda input: np.array_equal(input, x))
      batch.loc[con1, 'value'] = batch.loc[con1, 'value'] + ALPHA * (y - batch.loc[con1, 'value'])

    elif(state != None):
      #print(state)
      y = reward + self.gamma * self.getStateValue(next_state)
      #print(tf.get_static_value(y)[0], " y")
      x = np.array(state).flatten()

      if not (batch['state'].apply(lambda input: np.array_equal(input, x)).any()):
        batch.loc[len(batch.index)] = [x, 0]

      con1 = batch["state"].apply(lambda input: np.array_equal(input, x))
      batch.loc[con1, 'value'] = batch.loc[con1, 'value'] + ALPHA * (tf.get_static_value(y)[0] - batch.loc[con1, 'value'])

agent1 = Agent()
agent2 = Agent()

#@title Your Title Here
#NO BATCHING


def rotateArray(m):
  return list(zip(*m[::-1]))



batches = 50
episodes_per_batch = 20
epsilon = 0
board_size = 3
win_con = 3


for b in range(batches):
  print(b)
  batch = pd.DataFrame(columns = ["state", "value"])
  epsilon = 0.2 #1- (b/batches)
  for i in range(episodes_per_batch):
    print(i, " batch: ", b)
    game = Gomoku(board_size, win_con)
    winner = game.getWinner()
    agent1.last_state_action = None
    agent1.this_state_action = None
    agent2.last_state_action = None
    agent2.this_state_action = None
    r1 = 0
    r2 = 0

    while(winner == -1):
      #p1 play current estimate of best move
      legal_moves = game.getLegalMoves()

      best_move = None
      best_score = -1000
      for move in legal_moves:
        next = game.getNextState(move)
        value = agent1.getStateValue(next)
        if(value >= best_score):
          best_move = move
          best_score = value
      if (random.random() < epsilon):
        #print("random")
        best_move = random.choice(legal_moves)

      game.playMove(best_move)
      state_action_1_cur = game.getState()
      #print(state_action_1_cur[0])
      #print(state_action_1_cur[1])
      #print(state_action_1_cur[2])
      #print()
      agent1.updateState(state_action_1_cur)


      #update p1 with previous state action and reward
      currState = agent1.last_state_action
      nextState = agent1.this_state_action
      for i in range(0,3):
        agent1.updateBatch(currState, nextState, r1, batch)
        if(currState):
          currState = rotateArray(currState)
        if(nextState):
          nextState = rotateArray(nextState)
      #get reward r1
      r1 = game.getReward()

      #if p1 wins update agent for p2 with -r1 and update p1 with r1 and end
      #p1 will be updated with it's state action pair leading to an end state, as it's updates have always been one step behind.
      winner = game.getWinner()
      if(winner == 1 or winner == 0):
        currState=agent1.this_state_action
        for i in range(0,3):
          agent1.updateBatch(currState, None, r1, batch)
          currState = rotateArray(currState)
        
        currState=agent2.this_state_action
        for i in range(0,3):
          agent2.updateBatch(currState, None, -r1, batch)
          currState = rotateArray(currState)
        break


      #p2 play current estimate of best move
      legal_moves = game.getLegalMoves()

      best_move = None
      best_score = -1000
      for move in legal_moves:
        next = game.getNextState(move)
        value = agent1.getStateValue(next)
        if(value >= best_score):
          best_move = move
          best_score = value
      if (random.random() < epsilon):
        #print("random")
        best_move = random.choice(legal_moves)

      game.playMove(best_move)
      state_action_2_cur = game.getState()
      #print(state_action_2_cur[0])
      #print(state_action_2_cur[1])
      #print(state_action_2_cur[2])
      #print()
      agent2.updateState(state_action_2_cur)

      #update agent of p2 with previous state action and reward
      currState = agent2.last_state_action
      nextState = agent2.this_state_action
      for i in range(0,3):
        agent2.updateBatch(currState, nextState, r2, batch)
        if(currState):
          currState = rotateArray(currState)
        if(nextState):
          nextState = rotateArray(nextState)
      #get reward r2
      r2 = game.getReward()

      #if p2 wins update agent for p1 and p2 and end
      winner = game.getWinner()
      if(winner == 2 or winner == 0):
        currState=agent1.this_state_action
        for i in range(0,3):
          agent1.updateBatch(currState, None, r2, batch)
          currState = rotateArray(currState)
        
        currState=agent2.this_state_action
        for i in range(0,3):
          agent2.updateBatch(currState, None, -r2, batch)
          currState = rotateArray(currState)
        break

  #print(batch)
  agent1.updateAgentBatch(batch)
  #print(game.getState())
agent1.model.save("3x3ModelDanyuan1K.keras")
