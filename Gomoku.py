import numpy as np
import math

class Gomoku:
  def __init__(self, board_size = 19, win_con = 5):
    self.board_size = board_size
    self.win_con = win_con
    self.game = 0                #NOTE: current game state is stored as an integer
    self.last_move = -1
    self.cur_player = 1   #depricated

    self.__to_int_vector1 = np.array([3 ** i for i in range(self.board_size)])
    self.__to_int_vector2 = np.array([3 ** (i * self.board_size) for i in range(self.board_size)])

  #converts base 10 integer into base 3 integer into array (DON'T USE, THIS IS SLOW)
  def __toArray(self, input):
    array = []
    temp = input
    for i in range(self.board_size):
      row = []
      for j in range(self.board_size):
        row.append(temp % 3)
        temp = temp//3
      array.append(row)
    return array

  #converts array of game state into base 10 integer
  def __toInt(self, array):
    out = np.dot(np.dot(array, self.__to_int_vector1), self.__to_int_vector2)
    return out

  #DON'T USE, SLOW
  def getStateArray(self):
    return self.__toArray(self.game)
  
  def getState(self):
    return self.game

  def getNextState(self, move):
    return self.game + move
  
  def getPosValue(self, row, col):
    return ((self.game) // (self.__to_int_vector1[col] * self.__to_int_vector2[row])) % 3


  #returns 1 or 2 for the player that has won, 0 if the game has ended and is drawn, -1 if the game is not ended
    #returns winner of game, 0 if none
  def getWinner(self):
    #if no move has been made, return 0
    if(self.last_move == -1):
      return -1

    #initialization
    cur_state = self.game
    mover = 2 - (self.last_move % 2)                   #if last move is odd, then the player who moved was 1, if it was even then the player who moved was 2
    last_move = self.last_move/mover                   #reduces the mover to 
    cur_action = [0] * 2
    temp = int(math.log(last_move, 3) + 0.5)
    cur_action[1] = int(temp % self.board_size)     #log base 3 of the move equals col + (row * size)
    cur_action[0] = int(temp // self.board_size)
    

    #check vertical
    verticalCount=1
    currRowCheck=cur_action[0]+1
    #check upwards
    while currRowCheck<self.board_size and self.getPosValue(currRowCheck, cur_action[1])==mover:
        verticalCount+=1
        currRowCheck+=1

    currRowCheck=cur_action[0]-1
    #check downwards
    while currRowCheck>=0 and self.getPosValue(currRowCheck, cur_action[1])==mover:
        verticalCount+=1
        currRowCheck-=1

    if verticalCount>=self.win_con:
        return mover


    #check horizontal
    horizontalCount=1
    currColCheck=cur_action[1]+1

    #check right
    while currColCheck<self.board_size and self.getPosValue(cur_action[0],currColCheck)==mover:
        horizontalCount+=1
        currColCheck+=1

    currColCheck=cur_action[1]-1
    #check left
    while currColCheck>=0 and self.getPosValue(cur_action[0],currColCheck)==mover:
        horizontalCount+=1
        currColCheck-=1

    if horizontalCount>=self.win_con:
        return mover

    #check diagonal from top left to bot right
    topLeftDiagonalCount=1
    currRowCheck=cur_action[0]+1
    currColCheck=cur_action[1]+1

    #check down and to the right
    while currRowCheck<self.board_size and currColCheck<self.board_size and self.getPosValue(currRowCheck,currColCheck)==mover:
        topLeftDiagonalCount+=1
        currRowCheck+=1
        currColCheck+=1

    currRowCheck=cur_action[0]-1
    currColCheck=cur_action[1]-1

    #check up and to the left
    while currRowCheck>=0 and currColCheck>=0 and self.getPosValue(currRowCheck,currColCheck)==mover:
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
    while currRowCheck<self.board_size and currColCheck>=0 and self.getPosValue(currRowCheck,currColCheck)==mover:
        topRightDiagonalCount+=1
        currRowCheck+=1
        currColCheck-=1

    currRowCheck=cur_action[0]-1
    currColCheck=cur_action[1]+1

    #check up and to the right
    while currRowCheck>=0 and currColCheck<self.board_size and self.getPosValue(currRowCheck,currColCheck)==mover:
        topRightDiagonalCount+=1
        currRowCheck-=1
        currColCheck+=1

    if topRightDiagonalCount>=self.win_con:
        return mover

    legal_moves = self.getLegalMoves()
    if not legal_moves:
      return 0
    return -1

  #return list of legal moves in a the current state      #TODO make faster maybe
  def getLegalMoves(self):
    cur_state = self.game
    legal_moves = []
    move = 1
    for i in range(self.board_size):
      for j in range(self.board_size):
        if(cur_state % 3 == 0):
          legal_moves.append(move)
        cur_state = cur_state // 3
        move *= 3
    return legal_moves


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
    #print(cur_state)
    self.game = self.getNextState(move)
    self.last_move = move
    self.cur_player = ((self.cur_player) % 2) + 1
    #print(move)
    p = self.__toArray(self.game)
    #print(p[0])
    #print(p[1])
    #print(p[2])

