# track.py

import pygame
from utils import load_image
from config import Config


class Track:
    def __init__(self):
        self.image = load_image(Config.ASSETS_PATH, Config.TRACK_IMAGE)
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

        # Usar configuración de inicio desde Config
        self.start_position = Config.START_POSITION
        self.start_angle = Config.START_ANGLE

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def check_collision(self, car_mask, car_rect):
        """
        Comprueba si hay una colisión entre el coche y la pista.
        """
        offset = (car_rect.left - self.rect.left, car_rect.top - self.rect.top)
        collision_point = self.mask.overlap(car_mask, offset)
        return collision_point is not None  # Retorna True si hay colisión (área transparente)
