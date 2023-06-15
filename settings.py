# all the vatriables for the game
import pygame
from pygame.locals import *
from random import randrange

WINDOW_WIDTH = 620
WINDOW_HEIGHT = 720
TITLE = "Floppy Bird"
FPS = 60

# movement speeds
GRAVITY = .75
JUMP = 12

# colors
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
VERYLIGHTBLUE = (0, 150, 255)

# poles
POLE_WIDTH = 40
POLE_VEL = 4
POLE_CONST = 50
POLE_HEIGHT = randrange(0, WINDOW_WIDTH // 2)
POLE_PROLAZ = 200
POLE_DISTANCE = 220

# fonts
FONT = "arial"

# ai
DEATH_PENALTY = 5
CEILING_HIT_PENALTY = 1
FLOOR_HIT_PENALTY = 1
MOVING_REWARD = 0.1
PASSING_POLE_REWARD = 5

