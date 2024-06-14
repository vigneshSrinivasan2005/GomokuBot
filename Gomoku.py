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
