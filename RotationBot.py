import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import random
class Gomoku:
  def __init__(self, board_size = 19, win_con = 5):
    self.board_size = board_size
    self.win_con = win_con
    self.game = 0                #NOTE: current game state is stored as an integer
    self.last_move = -1
    self.cur_player = 1

  #converts base 10 integer into base 3 integer into array
  def __gameToArray(self):
    array = []
    temp = self.game
    for i in range(self.board_size):
      row = []
      for j in range(self.board_size):
        row.insert(0, temp % 3)
        temp = temp//3
      array.insert(0, row)
    return array

  #converts array of game state into base 10 integer
  def __gameToInt(self, array):
    base = 1
    integer = 0
    array_r = [array[i] for i in range(len(array) - 1, -1, -1)]
    for i in array_r:
      i_r = [i[k] for k in range(len(i) - 1, -1, -1)]
      for j in i_r:
        integer += j * base
        base = base * 3
    return integer

  #converts base 10 integer to array of 2 values with base = board size
  def __actionToArray(self, action):
    array = []
    temp = action
    for i in range(2):
      array.insert(0, temp % self.board_size)
      temp = temp//self.board_size
    return array

  #converts array of 2 values to base 10 int
  def __actionToInt(self, action):
    base = 1
    integer = 0
    action_r = [action[i] for i in range(len(action) - 1, -1, -1)]
    for i in action_r:
      integer += i * base
      base = base * self.board_size
    return integer

  def getState(self):
    return self.__gameToArray()

  def getNextState(self, move):
    s = self.__gameToArray()
    s[move[0]][move[1]] = self.cur_player
    return s


  #returns 1 or 2 for the player that has won, 0 if the game has ended and is drawn, -1 if the game is not ended
    #returns winner of game, 0 if none
  def getWinner(self):
    #if no move has been made, return 0
    if(self.last_move == -1):
      return -1

    #initialization
    cur_state = self.__gameToArray()
    cur_action = self.__actionToArray(self.last_move)
    cur_action[0] = int(cur_action[0])
    cur_action[1] = int(cur_action[1])
    mover = cur_state[cur_action[0]][cur_action[1]]

    #check vertical
    verticalCount=1
    currRowCheck=cur_action[0]+1
    #check upwards
    while currRowCheck<self.board_size and cur_state[currRowCheck][cur_action[1]]==mover:
        verticalCount+=1
        currRowCheck+=1

    currRowCheck=cur_action[0]-1
    #check downwards
    while currRowCheck>=0 and cur_state[currRowCheck][cur_action[1]]==mover:
        verticalCount+=1
        currRowCheck-=1

    if verticalCount>=self.win_con:
        return mover


    #check horizontal

    horizontalCount=1
    currColCheck=cur_action[1]+1

    #check right
    while currColCheck<self.board_size and cur_state[cur_action[0]][currColCheck]==mover:
        horizontalCount+=1
        currColCheck+=1

    currColCheck=cur_action[1]-1
    #check left
    while currColCheck>=0 and cur_state[cur_action[0]][currColCheck]==mover:
        horizontalCount+=1
        currColCheck-=1

    if horizontalCount>=self.win_con:
        return mover

    #check diagonal from top left to bot right
    topLeftDiagonalCount=1
    currRowCheck=cur_action[0]+1
    currColCheck=cur_action[1]+1

    #check down and to the right
    while currRowCheck<self.board_size and currColCheck<self.board_size and cur_state[currRowCheck][currColCheck]==mover:
        topLeftDiagonalCount+=1
        currRowCheck+=1
        currColCheck+=1

    currRowCheck=cur_action[0]-1
    currColCheck=cur_action[1]-1

    #check up and to the left
    while currRowCheck>=0 and currColCheck>=0 and cur_state[currRowCheck][currColCheck]==mover:
        topLeftDiagonalCount+=1
        currRowCheck-=1
        currColCheck-=1

    if topLeftDiagonalCount>=self.win_con:
        return mover

    #check diagonal from top right to bot left
    topRightDiagonalCount=1
    currRowCheck=cur_action[0]+1
    currColCheck=cur_action[1]-1

    #check down and to the left
    while currRowCheck<self.board_size and currColCheck>=0 and cur_state[currRowCheck][currColCheck]==mover:
        topRightDiagonalCount+=1
        currRowCheck+=1
        currColCheck-=1

    currRowCheck=cur_action[0]-1
    currColCheck=cur_action[1]+1

    #check up and to the right
    while currRowCheck>=0 and currColCheck<self.board_size and cur_state[currRowCheck][currColCheck]==mover:
        topRightDiagonalCount+=1
        currRowCheck-=1
        currColCheck+=1

    if topRightDiagonalCount>=self.win_con:
        return mover

    legal_moves = self.getLegalMoves()
    if not legal_moves:
      return 0
    return -1

  #return list of legal moves in a the current state
  def getLegalMoves(self):
    cur_state = self.__gameToArray()
    legal_moves = []
    for i in range(len(cur_state)):
      for j in range(len(cur_state[i])):
        if(cur_state[i][j] == 0):
          legal_moves.append([i, j])
    return legal_moves

  #returns the next state after a move from the current state
  def getNextState(self, move):
    state = self.__gameToArray()
    #print(str(move))
    state[move[0]][move[1]] = self.cur_player
    return state
  #returns reward from the current state
  def getReward(self):
    winner = self.getWinner()
    if(winner == -1):
      return 0
    if(winner == 0):
      return 0
    if(winner == 1):
      return 1
    if(winner == 2):
      return -1
  #updates internals of game
  def playMove(self, move):
    cur_state = self.__gameToArray()
    cur_state[move[0]][move[1]] = self.cur_player
    #print(cur_state)
    self.game = self.__gameToInt(cur_state)
    self.last_move = self.__actionToInt(move)
    self.cur_player = ((self.cur_player) % 2) + 1
    #print(move)
  
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
      return 0;
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
