import pygame
from pygame.locals import *
from settings import *
vec = pygame.math.Vector2

class Player(pygame.sprite.Sprite):

    def __init__(self, game):
        pygame.sprite.Sprite.__init__(self)
        self.game = game
        self.image = self.game.floppySprite[0]
        self.rect = self.image.get_rect()
        self.pos = vec(POLE_DISTANCE, WINDOW_HEIGHT // 4)
        self.rect.midbottom = self.pos
        self.ver_acc = GRAVITY
        self.ver_vel = 0
        self.alive = True

    def gravity(self):
        self.ver_acc = GRAVITY
        self.ver_vel += self.ver_acc
        self.pos.y += self.ver_vel

        self.rect.midbottom = self.pos

        if self.ver_vel > 20:
            self.ver_vel = 20

        if self.pos.y > WINDOW_HEIGHT + 20:
            self.pos.y = -10

    def jump(self):
        self.ver_vel = -JUMP


class Ground(pygame.sprite.Sprite):

    def __init__(self, x, y, w, h):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((w, h))
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect()
        self.pos = vec(x, y)
        self.rect.center = self.pos
        self.hor_vel = POLE_VEL

    def move(self):
        self.pos.x -= self.hor_vel
        self.rect.center = self.pos


class UpperPole(pygame.sprite.Sprite):

    def __init__(self, x, y, w, h):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((w, h))
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect()
        self.pos = vec(x, y)
        self.rect.midtop = self.pos
        self.hor_vel = POLE_VEL
        self.height = h


    def move(self):
        self.pos.x -= self.hor_vel
        self.rect.midtop = self.pos


class LowerPole(pygame.sprite.Sprite):

    def __init__(self, x, y, w, h):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((w, h))
        self.image.fill(YELLOW)
        self.rect = self.image.get_rect()
        self.pos = vec(x, y)
        self.rect.midbottom = self.pos
        self.hor_vel = POLE_VEL
        self.height = h


    def move(self):
        self.pos.x -= self.hor_vel
        self.rect.midbottom = self.pos


