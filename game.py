import pygame
from pygame.locals import *
from settings import *
from sprites import *
import os
import neat
import math


class Game:

    def __init__(self):

        # initializes pygame and creates a game window
        pygame.init()
        self.win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(TITLE)
        self.running = True
        self.clock = pygame.time.Clock()
        self.score = 0

        self.ground = Ground(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 10, WINDOW_WIDTH, 20)
        # self.floppy = None
        self.all_sprites = pygame.sprite.Group()
        self.poles = pygame.sprite.Group()

        # attributes for the AI
        self.floppies = None
        self.scores = None
        self.nets = None
        self.ge = None

        # loads the player sprites
        self.floppySprites = [pygame.image.load("Game_Assets/Floppy_Bird1.png").convert_alpha(),
                              pygame.image.load("Game_Assets/Floppy_Bird2.png").convert_alpha(),
                              pygame.image.load("Game_Assets/Floppy_Bird3.png").convert_alpha(),
                              pygame.image.load("Game_Assets/Floppy_Bird4.png").convert_alpha()]

        # changes the game window icon
        pygame.display.set_icon(self.floppySprites[0])

        # animation settings
        self.anim = True

    def init_floppies(self, floppies, nets, ge):
        if (len(floppies) != len(nets) and 
            len(floppies) != len(ge) and 
            len(nets) != len(ge)):
            raise Exception("All lists must have the same number elements!")
        self.floppies = floppies
        self.nets = nets
        self.ge = ge
        self.scores = [0 for _ in range(len(self.floppies))]

    def events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
                if self.playing:
                    self.playing = False
            if event.type == KEYDOWN:
                pass
                # for floppy in self.floppies:
                #     if event.key == K_SPACE:
                #         if floppy.alive:
                #             floppy.jump()

                # if event.key == K_SPACE:
                #     if self.floppy.alive:
                #         self.floppy.jump()

                # if event.key == K_LSHIFT:
                #     self.playing = False
                #     self.floppy.alive = True
                #     self.scores = [0 for _ in range(len(self.floppies))]
                #     self.floppy.anim = True
                #     self.anim = True

    def run(self):
        self.playing = True
        while self.playing:
            self.clock.tick(FPS)
            self.updateSprites()
            self.events()
            self.drawSprites()

    def newGame(self):
        self.all_sprites = pygame.sprite.Group()
        self.poles = pygame.sprite.Group()
        POLE_X = WINDOW_WIDTH
        for _ in range(3):
            POLE_height = randrange(0, WINDOW_HEIGHT - POLE_PROLAZ)
            p1 = UpperPole(POLE_X, 0, 40, POLE_height)
            p2 = LowerPole(POLE_X, WINDOW_HEIGHT, 40, WINDOW_HEIGHT - POLE_height - POLE_PROLAZ)
            self.poles.add(p1)
            self.poles.add(p2)
            self.all_sprites.add(p1)
            self.all_sprites.add(p2)
            POLE_X += POLE_DISTANCE
        # self.floppy = Player(self)
        self.all_sprites.add(self.ground)
        # self.all_sprites.add(self.floppy)
        for floppy in self.floppies:
            self.all_sprites.add(floppy)
        self.run()

    def are_all_dead(self):
        all_dead = True
        for floppy in self.floppies:
            all_dead = all_dead and not floppy.alive
        return all_dead

    def get_closest_pole(self, pole_type):
        floppy = None
        for f in self.floppies:
            if f.alive:
                floppy = f
                break

        closest_pole = None
        closest_distance = math.inf
        if floppy is not None:
            for pole in self.poles:
                if (type(pole) == pole_type):
                    if floppy.pos.x < pole.pos.x:
                        distance = pole.pos.x - floppy.pos.x
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_pole = pole

        return closest_pole

    def drawSprites(self):
        self.win.fill(VERYLIGHTBLUE)
        self.all_sprites.draw(self.win)
        self.drawText(FONT, 48, str(round(self.score)), WINDOW_WIDTH // 2, WINDOW_HEIGHT // 8, (0, 0, 0))

        if self.are_all_dead():
            # for floppy in self.floppies:
            #     floppy.alive = True
            #     floppy.pos.y = WINDOW_HEIGHT // 4
            # self.newGame()
            self.playing = False
            self.running = False

        # if self.floppy.alive == False:
        #     self.showGameOverScreen()
        #     self.anim = False

        pygame.display.flip()

    def updateSprites(self):
        self.all_sprites.update()
        
        self.animate()
        for pole in self.poles:
            pole.move()

        any_floppy_scored = False
        for i, floppy in enumerate(self.floppies):
            floppy.gravity()
            if floppy.alive:
                self.ge[i].fitness += MOVING_REWARD
                closest_upper_pole = self.get_closest_pole(UpperPole)
                closest_lower_pole = self.get_closest_pole(LowerPole)
                output = self.nets[i].activate((floppy.pos.y, abs(floppy.pos.y - closest_upper_pole.rect.bottom), abs(floppy.pos.y - closest_lower_pole.rect.top)))

                if output[0] > 0.5:
                    floppy.jump()

            hits = pygame.sprite.spritecollide(floppy, self.poles, False)
            for pole in self.poles:
                floppy_collided = False
                if hits:
                    floppy_collided = True
                if floppy.pos.y >= self.ground.rect.top:
                    floppy.pos.y = self.ground.rect.top
                    floppy.ver_vel = 0
                    floppy_collided = True
                if floppy.pos.y <= 32:
                    floppy.pos.y = 32
                    floppy_collided = True

                if floppy_collided:
                    self.ge[i].fitness -= DEATH_PENALTY 
                    floppy.alive = False
                    self.all_sprites.remove(floppy)
                    # self.floppies.pop(i)
                    # self.ge.pop(i)
                    # self.nets.pop(i)

            # deletes and creates more poles based on their onscreen position also checks and changes the score
            for pole in self.poles:
                if pole.rect.right <= 0:
                    pole.kill()
                if floppy.pos.x == pole.pos.x:
                    if floppy.alive:
                        any_floppy_scored = True
                        self.ge[i].fitness += PASSING_POLE_REWARD

        if any_floppy_scored:
            self.score += 1
        # self.floppy.gravity()

        # detects collision between the player and game objects
        # hits = pygame.sprite.spritecollide(self.floppy, self.poles, False)
        # for pole in self.poles:
        #     if hits:
        #         pole.hor_vel = 0
        #         self.floppy.alive = False
        #     if self.floppy.pos.y >= self.ground.rect.top:
        #         self.floppy.pos.y = self.ground.rect.top
        #         self.floppy.ver_vel = 0
        #         pole.hor_vel = 0
        #         self.floppy.alive = False
        #     if self.floppy.pos.y <= 32:
        #         self.floppy.pos.y = 32
        #         pole.hor_vel = 0
        #         self.floppy.alive = False

        # deletes and creates more poles based on their onscreen position also checks and changes the score
        # for pole in self.poles:
        #     if pole.rect.right <= 0:
        #         pole.kill()
        #     if self.floppy.pos.x == pole.pos.x:
        #         if self.floppy.alive:
        #             self.score += .5

        while len(self.poles) < 6:
            POLE_height = randrange(0, WINDOW_HEIGHT - POLE_PROLAZ)
            p1 = UpperPole(WINDOW_WIDTH + 20, 0, 40, POLE_height)
            p2 = LowerPole(WINDOW_WIDTH + 20, WINDOW_HEIGHT, 40, WINDOW_HEIGHT - POLE_height - POLE_PROLAZ)
            self.poles.add(p1)
            self.poles.add(p2)
            self.all_sprites.add(p1)
            self.all_sprites.add(p2)

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

        for floppy in self.floppies:
            if floppy.anim_count > 12:
                floppy.anim_count = 0
            floppy.image = self.floppySprites[floppy.anim_count % 4]
            floppy.anim_count += 1

        # if self.anim:
        #     self.floppy.image = self.floppySprite[self.anim_count//4]
        #     self.anim_count += 1


def main(genomes, config):
    game = Game()

    nets = []
    ge = []
    floppies = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        floppies.append(Player(game.floppySprites[0]))
        g.fitness = 0
        ge.append(g)

    game.init_floppies(floppies, nets, ge)

    while game.running:
        game.newGame()
    
    print()
    print("--------------------------")
    print(f"Final score: {game.score}")
    print("--------------------------")
    print()


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
     
    winner = p.run(main, 100)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)
