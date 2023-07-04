import math
import pygame
from pygame.locals import *
import os
import sys
path = os.path.abspath("")
sys.path.append(path)
from settings import *
from sprites2 import *
import numpy as np
from pygame.surfarray import array3d, pixels_alpha


class Game:

    def __init__(self):

        # initializes pygame and creates a game window
        pygame.init()
        self.win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.running = True
        self.clock = pygame.time.Clock()
        self.score = 0
        self.reward = 0

        # loads the player sprites
        self.floppySprite = [pygame.image.load("Game_Assets/Floppy_Bird1.png").convert_alpha(),
                             pygame.image.load("Game_Assets/Floppy_Bird2.png").convert_alpha(),
                             pygame.image.load("Game_Assets/Floppy_Bird3.png").convert_alpha(),
                             pygame.image.load("Game_Assets/Floppy_Bird4.png").convert_alpha()]

        # changes the game window icon
        pygame.display.set_icon(self.floppySprite[0])

        # animation settings
        self.floppy = Player(self)
        self.anim_count = 0
        self.anim = True
        self.playing = True
        self.floppy.alive = True
        self.score = 0
        self.floppy.anim = True
        self.anim = True

        self.all_sprites = pygame.sprite.Group()
        self.poles = pygame.sprite.Group()
        POLE_X = WINDOW_WIDTH
        for _ in range(3):
            POLE_height = randrange(0, WINDOW_HEIGHT - POLE_GAP)
            p1 = UpperPole(POLE_X, 0, POLE_WIDTH, POLE_height)
            p2 = LowerPole(POLE_X, WINDOW_HEIGHT, POLE_WIDTH, WINDOW_HEIGHT - POLE_height - POLE_GAP)
            self.poles.add(p1)
            self.poles.add(p2)
            self.all_sprites.add(p1)
            self.all_sprites.add(p2)
            POLE_X += POLE_DISTANCE
        self.ground = Ground(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 10, WINDOW_WIDTH, 20)
        # self.floppy = Player(self)
        self.all_sprites.add(self.ground)
        self.all_sprites.add(self.floppy)

    def events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
                if self.playing:
                    self.playing = False

    
    def nextFrame(self, action):
        self.clock.tick(FPS)
        new_obs, rew, done, score = self.updateSprites(action)
        self.drawSprites()
        return new_obs, rew, done, score

    def drawSprites(self):
        self.win.fill(VERYLIGHTBLUE)
        self.all_sprites.draw(self.win)
        self.drawText(FONT, 48, str(round(self.score)), WINDOW_WIDTH // 2, WINDOW_HEIGHT // 8, (0, 0, 0))
        if self.floppy.alive == False:
            self.showGameOverScreen()
            self.anim = False

        pygame.display.flip()

    def updateSprites(self, action):
        self.reward = 0.1
        if self.floppy.alive and action==1:
            self.floppy.jump()
        self.all_sprites.update()
        self.floppy.gravity()
        self.animate()
        for pole in self.poles:
            pole.move()

        # detects collision between the player and game objects
        hits = pygame.sprite.spritecollide(self.floppy, self.poles, False)
        for pole in self.poles:
            if hits:
                pole.hor_vel = 0
                self.floppy.alive = False
            if self.floppy.pos.y >= self.ground.rect.top:
                self.floppy.pos.y = self.ground.rect.top
                self.floppy.ver_vel = 0
                pole.hor_vel = 0
                self.floppy.alive = False
            if self.floppy.pos.y <= 32:
                self.floppy.pos.y = 32
                pole.hor_vel = 0
                self.floppy.alive = False
        if not self.floppy.alive:
            self.reward = -1

        # deletes and creates more poles based on their onscreen position also checks and changes the score
        for pole in self.poles:
            if pole.rect.right <= 0:
                pole.kill()
            if self.floppy.pos.x == pole.pos.x:
                if self.floppy.alive:
                    self.score += .5
                    self.reward = 1
        while len(self.poles) < 6:
            POLE_height = randrange(0, WINDOW_HEIGHT - POLE_GAP)
            p1 = UpperPole(WINDOW_WIDTH + 20, 0, POLE_WIDTH, POLE_height)
            p2 = LowerPole(WINDOW_WIDTH + 20, WINDOW_HEIGHT, POLE_WIDTH, WINDOW_HEIGHT - POLE_height - POLE_GAP)
            self.poles.add(p1)
            self.poles.add(p2)
            self.all_sprites.add(p1)
            self.all_sprites.add(p2)
        pole_x, pole_y = self.get_closest_pole()
        # image = array3d(pygame.display.get_surface())
        # return image, self.reward, not self.floppy.alive, self.score
        return np.array([round(self.floppy.pos.x), round(self.floppy.pos.y), round(pole_x), round(pole_y)], dtype=np.float32), self.reward, not self.floppy.alive, self.score

    def showTitleScreen(self):
        self.win.fill(VERYLIGHTBLUE)
        self.drawText(FONT, 48, "Floppy Bird", WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, (0, 0, 0))
        self.drawText(FONT, 32, "Press <LSHIFT> to play", WINDOW_WIDTH // 2, WINDOW_HEIGHT - WINDOW_HEIGHT // 3,
                      (0, 0, 0))
        pygame.display.flip()
        self.wait()

    def showGameOverScreen(self):
        self.drawText(FONT, 48, "GAME OVER", WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2, (0, 0, 0))
        self.drawText(FONT, 32, "Press <LSHIFT> to play again", WINDOW_WIDTH // 2, WINDOW_HEIGHT - WINDOW_HEIGHT // 3, (0, 0, 0))

    def drawText(self, font, size, text, x, y, color):
        font = pygame.font.SysFont(font, size)
        text = font.render(text, 1, color)
        text_rect = text.get_rect()
        text_rect.center = (x, y)
        self.win.blit(text, text_rect)

    def wait(self):
        self.waiting = True
        while self.waiting:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_LSHIFT:
                        self.waiting = False
                if event.type == QUIT:
                    self.waiting = False
                    self.running = False

    def animate(self):
        if self.anim_count > 12:
            self.anim_count = 0

        if self.anim:
            self.floppy.image = self.floppySprite[self.anim_count//4]
            self.anim_count += 1

    def get_closest_pole(self):
        closest_pole = None
        closest_distance = math.inf
        if self.floppy.alive:
            for pole in self.poles:
                if type(pole) == LowerPole and self.floppy.pos.x <= pole.pos.x:
                    distance = pole.pos.x - self.floppy.pos.x
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_pole = pole
        if closest_pole is None:
            return 0, 0
        return closest_pole.pos.x - self.floppy.pos.x, closest_pole.rect.top - self.floppy.pos.y
    
def skip_frames(game, x):
    for _ in range(x):
        new_obs, rew, done, score = game.nextFrame(0)
    return new_obs, rew, done, score

if __name__ == "__main__":
    print(int(1e6))
    # game = Game()

    # #game.showTitleScreen()
    # # while game.floppy.alive:
    # # game.nextFrame(1)
    # # game.nextFrame(0)
    # # game.nextFrame(0)
    # # game.nextFrame(0)
    # # game.nextFrame(0)
    # # game.nextFrame(1)
    # # game.nextFrame(0)


    game = Game()
    # #game.showTitleScreen()
    # # while game.floppy.alive:
    print(game.nextFrame(0))
    skip_frames(game, 5)
    new_obs, rew, done, _ = game.nextFrame(0)
    print(type(new_obs))
    skip_frames(game, 5)
    print(game.nextFrame(1))
    skip_frames(game, 5)
    print(game.nextFrame(0))
    skip_frames(game, 5)
    print(game.nextFrame(0))
    skip_frames(game, 5)
    print(game.nextFrame(0))
    skip_frames(game, 5)
    game.nextFrame(0)
    skip_frames(game, 5)
    game.nextFrame(0)
    skip_frames(game, 5)
    game.nextFrame(1)
    # skip_frames(game)
    # game.nextFrame(1)
    # skip_frames(game)
    # game.nextFrame(1)
    # skip_frames(game)
    # game.nextFrame(1)
    # skip_frames(game)
    # game.nextFrame(1)
    # skip_frames(game)
    # game.nextFrame(1)
    # skip_frames(game)
    # game.nextFrame(1)
    # skip_frames(game)

    # game = Game()
    # #game.showTitleScreen()
    # while game.running:
    #     game.newGame(0)
    
        

