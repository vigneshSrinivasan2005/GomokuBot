from Agent import Agent
import pygame as pg
import sys
import time
from pygame.locals import *
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
    screen = pg.display.set_mode((width, height + 100), 0, 32)
    
    # setting up a nametag for the
    # game window
    pg.display.set_caption("My Tic Tac Toe")
    
    # loading the images as python object
    initiating_window = pg.image.load("modified_cover.png")
    x_img = pg.image.load("X_modified.png")
    y_img = pg.image.load("o_modified.png")
    
    # resizing images
    initiating_window = pg.transform.scale(
        initiating_window, (width, height + 100))
    x_img = pg.transform.scale(x_img, (80, 80))
    o_img = pg.transform.scale(y_img, (80, 80))
 
    def __init__(self, player, alpha, input_model=None):
        super().__init__(player, alpha, 3)
        if input_model != None:
            self.model = input_model
        Player.game_initiating_window()
        
        
    @staticmethod
    def game_initiating_window():
    
        # displaying over the screen
        Player.screen.blit( Player.initiating_window, (0, 0))
    
        # updating the display
        pg.display.update()
        time.sleep(3)
        Player.screen.fill( Player.white)
    
        # drawing vertical lines
        pg.draw.line( Player.screen,  Player.line_color, ( Player.width / 3, 0), ( Player.width / 3,  Player.height), 7)
        pg.draw.line( Player.screen,  Player.line_color, ( Player.width / 3 * 2, 0),
                    ( Player.width / 3 * 2,  Player.height), 7)
    
        # drawing horizontal lines
        pg.draw.line( Player.screen,  Player.line_color, (0,  Player.height / 3), ( Player.width,  Player.height / 3), 7)
        pg.draw.line( Player.screen,  Player.line_color, (0,  Player.height / 3 * 2),
                    ( Player.width,  Player.height / 3 * 2), 7)
        
    
    @staticmethod
    def drawXO(row, col):
        global board, XO
    
        # for the first row, the image
        # should be pasted at a x coordinate
        # of 30 from the left margin
        if row == 1:
            posx = 30
    
        # for the second row, the image
        # should be pasted at a x coordinate
        # of 30 from the game line
        if row == 2:
    
            # margin or width / 3 + 30 from
            # the left margin of the window
            posx = width / 3 + 30
    
        if row == 3:
            posx = width / 3 * 2 + 30
    
        if col == 1:
            posy = 30
    
        if col == 2:
            posy = height / 3 + 30
    
        if col == 3:
            posy = height / 3 * 2 + 30
    
        # setting up the required board
        # value to display
        board[row-1][col-1] = XO
    
        if(XO == 'x'):
    
            # pasting x_img over the screen
            # at a coordinate position of
            # (pos_y, posx) defined in the
            # above code
            screen.blit(x_img, (posy, posx))
            XO = 'o'
    
        else:
            screen.blit(o_img, (posy, posx))
            XO = 'x'
        pg.display.update()
    
    
    def user_click():
        # get coordinates of mouse click
        x, y = pg.mouse.get_pos()
    
        # get column of mouse click (1-3)
        if(x < width / 3):
            col = 1
    
        elif (x < width / 3 * 2):
            col = 2
    
        elif(x < width):
            col = 3
    
        else:
            col = None
    
        # get row of mouse click (1-3)
        if(y < height / 3):
            row = 1
    
        elif (y < height / 3 * 2):
            row = 2
    
        elif(y < height):
            row = 3
    
        else:
            row = None
    
        # after getting the row and col,
        # we need to draw the images at
        # the desired positions
        if(row and col and board[row-1][col-1] is None):
            global XO
    
            drawXO(row, col)
            check_win()
    

    def getMove(self, state, epsilon):


        return best_move

    def update(self):
        return 

    def updateBatch(self, reward):
        return
    
    def save(self, name):
        return