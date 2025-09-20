# game.py

import pygame
from config import Config
from car import Car
from track import Track


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT))
        pygame.display.set_caption("Simulación de Coche")
        self.clock = pygame.time.Clock()
        self.running = True

        # Crear la pista y el coche
        self.track = Track()
        self.car = Car(self.track)  # Pasar la pista al coche

    def run(self):
        while self.running:
            dt = self.clock.tick(Config.FPS) / 1000  # Delta time en segundos
            self.handle_events()
            self.update(dt)
            self.draw()
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self, dt):
        self.car.update(dt, self.track)

    def draw(self):
        self.screen.fill((255, 255, 255))  # Fondo blanco
        self.track.draw(self.screen)
        self.car.draw(self.screen)
        pygame.display.flip()