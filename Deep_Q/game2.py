import pygame
from pygame.locals import *
import os
import sys
path = os.path.abspath("")
sys.path.append(path)
from settings import *
from sprites2 import *
from pygame.surfarray import array3d, pixels_alpha
from pygame import display, time, init, Rect

class Game:

    def __init__(self):

        # initializes pygame and creates a game window
        pygame.init()
        self.win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        # pygame.display.set_caption(TITLE)
        self.running = True
        self.clock = pygame.time.Clock()
        self.score = 0
        self.reward = 0 # ima vrednosti 1 ako prode iymedju sipki, -1 ako umre a 0.1 ako samo zivi

        # loads the player sprites
        self.floppySprite = [pygame.image.load("Game_Assets/Floppy_Bird1.png").convert_alpha(),
                             pygame.image.load("Game_Assets/Floppy_Bird2.png").convert_alpha(),
                             pygame.image.load("Game_Assets/Floppy_Bird3.png").convert_alpha(),
                             pygame.image.load("Game_Assets/Floppy_Bird4.png").convert_alpha()]

        # changes the game window icon
        # pygame.display.set_icon(self.floppySprite[0])

        # animation settings
        self.anim_count = 0
        self.anim = True

    def events(self):
        pass
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
                if self.playing:
                    self.playing = False
            if event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.floppy.notDead:
                        self.floppy.jump()
                if self.floppy.notDead == False:
                    if event.key == K_LSHIFT:
                        self.playing = False
                        self.floppy.notDead = True
                        self.score = 0
                        self.floppy.anim = True
                        self.anim = True

    def run(self):
        self.playing = True
        # while self.playing:
        self.clock.tick(FPS)
        self.updateSprites()
            # self.events()
            # self.drawSprites()

    def newGame(self):
        self.all_sprites = pygame.sprite.Group()
        self.poles = pygame.sprite.Group()
        POLE_X = WINDOW_WIDTH
        for i in range(3):
            POLE_height = randrange(0, WINDOW_HEIGHT - POLE_GAP)
            p1 = UpperPole(POLE_X, 0, 40, POLE_height)
            p2 = LowerPole(POLE_X, WINDOW_HEIGHT, 40, WINDOW_HEIGHT - POLE_height - POLE_GAP)
            self.poles.add(p1)
            self.poles.add(p2)
            self.all_sprites.add(p1)
            self.all_sprites.add(p2)
            POLE_X += POLE_DISTANCE
        self.ground = Ground(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 10, WINDOW_WIDTH, 20)
        self.floppy = Player(self)
        self.all_sprites.add(self.ground)
        self.all_sprites.add(self.floppy)
        # self.run()

    def drawSprites(self):
        self.win.fill(VERYLIGHTBLUE)
        self.all_sprites.draw(self.win)
        self.drawText(FONT, 48, str(round(self.score)), WINDOW_WIDTH // 2, WINDOW_HEIGHT // 8, (0, 0, 0))
        if self.floppy.notDead == False:
            self.showGameOverScreen()
            self.anim = False

       

    def updateSprites(self, action=0):
        self.clock.tick(FPS)
        if action == 1:
            self.floppy.jump()

        self.reward = 0.1
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
                self.floppy.notDead = False
            if self.floppy.pos.y >= self.ground.rect.top:
                self.floppy.pos.y = self.ground.rect.top
                self.floppy.ver_vel = 0
                pole.hor_vel = 0
                self.floppy.notDead = False
            if self.floppy.pos.y <= 32:
                self.floppy.pos.y = 32
                pole.hor_vel = 0
                self.floppy.notDead = False

        if not self.floppy.notDead:
            self.reward = -1

        # deletes and creates more poles based on their onscreen position also checks and changes the score
        for pole in self.poles:
            if pole.rect.right <= 0:
                pole.kill()
            if self.floppy.pos.x == pole.pos.x:
                if self.floppy.notDead:
                    self.score += .5
                    self.reward = 1
        while len(self.poles) < 6:
            POLE_height = randrange(0, WINDOW_HEIGHT - POLE_GAP)
            p1 = UpperPole(WINDOW_WIDTH + 20, 0, 40, POLE_height)
            p2 = LowerPole(WINDOW_WIDTH + 20, WINDOW_HEIGHT, 40, WINDOW_HEIGHT - POLE_height - POLE_GAP)
            self.poles.add(p1)
            self.poles.add(p2)
            self.all_sprites.add(p1)
            self.all_sprites.add(p2)

        image = array3d(display.get_surface())
        return image, self.reward, self.floppy.notDead

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


if __name__ == "__main__":
    game = Game()
    game.showTitleScreen()
    while game.running:
        game.newGame()
