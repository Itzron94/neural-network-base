# car.py

import pygame
import math
from utils import load_image, rotate_image
from config import Config


class Car:
    def __init__(self, track):
        # Escalar la imagen del coche usando Config.CAR_SCALE
        original_image = load_image(Config.ASSETS_PATH, Config.CAR_IMAGE)
        scaled_width = int(original_image.get_width() * Config.CAR_SCALE)
        scaled_height = int(original_image.get_height() * Config.CAR_SCALE)
        self.original_image = pygame.transform.scale(original_image, (scaled_width, scaled_height))

        # Usar posición y ángulo inicial desde la pista
        self.image = self.original_image
        self.position = pygame.math.Vector2(track.start_position)
        self.angle = track.start_angle
        self.rect = self.image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.image)
        self.velocity = pygame.math.Vector2(0, 0)
        self.speed = 0

    def handle_input(self, dt):
        keys = pygame.key.get_pressed()
        acceleration = 0
        steering = 0

        if keys[pygame.K_UP]:
            acceleration = Config.ACCELERATION
        elif keys[pygame.K_DOWN]:
            acceleration = -Config.ACCELERATION

        if keys[pygame.K_LEFT]:
            steering = -Config.STEERING_SPEED
        elif keys[pygame.K_RIGHT]:
            steering = Config.STEERING_SPEED

        # Actualizar velocidad
        if acceleration != 0:
            self.speed += acceleration * dt
            self.speed = max(-Config.MAX_SPEED, min(self.speed, Config.MAX_SPEED))
        else:
            # Aplicar desaceleración natural
            if self.speed > 0:
                self.speed = max(self.speed - Config.DECELERATION * dt, 0)
            elif self.speed < 0:
                self.speed = min(self.speed + Config.DECELERATION * dt, 0)

        # Actualizar ángulo solo si el coche está en movimiento
        if steering != 0 and self.speed != 0:
            rotation_amount = (steering * dt) * (self.speed / Config.MAX_SPEED)
            self.angle += rotation_amount

        # Actualizar posición
        radians = math.radians(self.angle)
        self.velocity.x = math.sin(radians) * self.speed * dt
        self.velocity.y = -math.cos(radians) * self.speed * dt  # Invertir Y
        self.position += self.velocity

        # Actualizar imagen y rectángulo
        self.image = rotate_image(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.position)
        self.mask = pygame.mask.from_surface(self.image)

    def update(self, dt, track):
        self.handle_input(dt)

        # Verificar colisiones
        if track.check_collision(self.mask, self.rect):
            print("¡Colisión detectada!")
            # Detener el coche y retroceder un poco
            self.speed = 0
            self.position -= self.velocity * 2  # Retrocede para salir de la colisión
            self.rect = self.image.get_rect(center=self.position)
            self.mask = pygame.mask.from_surface(self.image)

    def draw(self, surface):
        surface.blit(self.image, self.rect)

    def apply_action(self, acceleration, steering):
        """
        Aplica la acción recibida (aceleración y dirección).
        """
        if acceleration > 0:
            self.speed += Config.ACCELERATION / 60  # Supone 60 FPS
        elif acceleration < 0:
            self.speed -= Config.ACCELERATION / 60

        # Limitar la velocidad
        self.speed = max(-Config.MAX_SPEED, min(self.speed, Config.MAX_SPEED))

        # Girar el coche
        if steering != 0 and self.speed != 0:
            self.angle += steering * (self.speed / Config.MAX_SPEED)
