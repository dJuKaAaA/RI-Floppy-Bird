# all the vatriables for the game
import pygame
from pygame.locals import *
from random import randrange

WINDOW_WIDTH = 620
WINDOW_HEIGHT = 720
TITLE = "Floppy Bird"
FPS = 60
FLOPPY_START_X_COORD = 250

# movement speeds
GRAVITY = .75
JUMP = 12

# colors
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
VERYLIGHTBLUE = (0, 150, 255)

# poles
POLE_WIDTH = 28
POLE_VEL = 4
POLE_CONST = 50
POLE_HEIGHT = randrange(0, WINDOW_WIDTH // 2)
POLE_GAP = 250
POLE_DISTANCE = 250

# fonts
FONT = "arial"

# ai
MOVING_REWARD = 1 / FPS
DEATH_PENALTY = 10
PASSING_POLE_REWARD = 5

