from Agent import Agent
import pygame as pg
import sys
import time
from pygame.locals import *
import numpy as np
class Player(Agent):

    # to set width of the game window
    width = 400
    
    # to set height of the game window
    height = 400
    
    # to set background color of the
    # game window
    white = (255, 255, 255)
    
    # color of the straightlines on that
    # white game board, dividing board
    # into 9 parts
    line_color = (0, 0, 0)
 
    screen = pg.display.set_mode((1280, 720))
    clock = pg.time.Clock()
    running = True
    # initializing the pygame window
    pg.init()
    
    # setting fps manually
    fps = 30
    
    # this is used to track time
    CLOCK = pg.time.Clock()
    
    # this method is used to build the
    # infrastructure of the display
    screen = pg.display.set_mode((width, height), 0, 32)
    
    # setting up a nametag for the
    # game window
    pg.display.set_caption("My Tic Tac Toe")
    
    # loading the images as python object
    x_img = pg.image.load("./Assets/X_modified.png")
    y_img = pg.image.load("./Assets/o_modified.png")
    
    # resizing images
    x_img = pg.transform.scale(x_img, (80, 80))
    o_img = pg.transform.scale(y_img, (80, 80))
 
    def __init__(self, player):
        super().__init__(player, USEREVENT_DROPFILE, 3)
        Player.game_initiating_window()
        
        
    @staticmethod
    def game_initiating_window():
        
        # updating the display
        Player.screen.fill( Player.white)
    
        # drawing vertical lines
        pg.draw.line( Player.screen,  Player.line_color, ( Player.width / 3, 0), ( Player.width / 3,  Player.height), 7)
        pg.draw.line( Player.screen,  Player.line_color, ( Player.width / 3 * 2, 0),
                    ( Player.width / 3 * 2,  Player.height), 7)
    
        # drawing horizontal lines
        pg.draw.line( Player.screen,  Player.line_color, (0,  Player.height / 3), ( Player.width,  Player.height / 3), 7)
        pg.draw.line( Player.screen,  Player.line_color, (0,  Player.height / 3 * 2),
                    ( Player.width,  Player.height / 3 * 2), 7)
        pg.display.update()

        
    
    def drawBoard(state):
        Player.game_initiating_window()
        rows=len(state)
        cols=len(state[0])
        for row in range(rows):
            for col in range(cols):
                if state[row][col] != 0:
                    posx= 30 + Player.width / cols * (col)
                    posy = 30 + Player.height / rows * (row)
                    XO= 'x' if state[row][col]==1 else 'o'
                    if(XO == 'x'):
                        # pasting x_img over the screen
                        # at a coordinate position of
                        # (pos_y, posx) defined in the
                        # above code
                        Player.screen.blit(Player.x_img, (posx, posy))    
                    else:
                        Player.screen.blit(Player.o_img, (posx, posy))
        pg.display.update()
    
    
    def user_click(board,player):
        # get coordinates of mouse click
        x, y = pg.mouse.get_pos()
        boardSize=len(board)

        col= int(x // (Player.width / boardSize))
        row= int(y // (Player.height / boardSize))
        # check if full
        if board[row][col] != 0:
            return None
        move=np.zeros((boardSize,boardSize)).tolist()
        move[row][col]=player
        return move

    def getMove(self, state, epsilon):

        state=super()._toArray(state)
        Player.drawBoard(state)
        pg.display.update()

        move=None
        while move is None:
            if pg.event.get(MOUSEBUTTONDOWN):
                move=Player.user_click(state,self.player,)
                if move is not None:
                    move=super()._toInt(Player.user_click(state,self.player,))
        return move

    def update(self):
        return 

    def updateBatch(self, reward):
        return
    
    def save(self, name):
        return