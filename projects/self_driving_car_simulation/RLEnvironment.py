# rl_environment.py

import numpy as np
from car import Car
from track import Track
from config import Config


class RLEnvironment:
    def __init__(self):
        self.track = Track()
        self.car = Car(self.track)
        self.done = False

    def reset(self):
        """
        Reinicia el entorno y devuelve el estado inicial.
        """
        self.car = Car(self.track)  # Reinicia el coche
        self.done = False
        return self._get_state()

    def step(self, action):
        """
        Aplica una acción, actualiza el entorno y devuelve:
        - El nuevo estado
        - La recompensa
        - Si el episodio ha terminado
        """
        # Desempaquetar la acción (acelerar, girar)
        acceleration, steering = action

        # Aplicar la acción al coche
        self.car.apply_action(acceleration, steering)

        # Actualizar el entorno
        self.car.update(1 / 60, self.track)  # Suponemos 60 FPS

        # Obtener estado y calcular recompensa
        state = self._get_state()
        reward, self.done = self._calculate_reward()

        return state, reward, self.done

    def _get_state(self):
        """
        Devuelve el estado actual del entorno como un vector de características.
        """
        # Observaciones: posición, velocidad, ángulo, distancias a bordes
        position = self.car.position
        velocity = self.car.speed
        angle = self.car.angle

        # Distancias a los bordes (simular sensores)
        distances = self._get_distances_to_edges()

        return np.array([position.x, position.y, velocity, angle] + distances)

    def _get_distances_to_edges(self):
        """
        Simula sensores de distancia que detectan los bordes de la pista.
        """
        distances = []
        directions = [
            (0, -1),  # Adelante
            (1, -1),  # Diagonal derecha
            (1, 0),  # Derecha
            (1, 1),  # Diagonal trasera derecha
            (0, 1),  # Atrás
            (-1, 1),  # Diagonal trasera izquierda
            (-1, 0),  # Izquierda
            (-1, -1)  # Diagonal izquierda
        ]

        for direction in directions:
            distance = self._raycast_distance(direction)
            distances.append(distance)

        return distances

    def _raycast_distance(self, direction):
        """
        Simula un sensor que mide la distancia al borde en una dirección dada.
        """
        position = np.array(self.car.position, dtype=np.int32)
        direction = np.array(direction)
        distance = 0

        while True:
            position += direction
            distance += 1

            # Verificar si la posición está fuera de los límites
            if not (0 <= position[0] < self.track.rect.width and 0 <= position[1] < self.track.rect.height):
                break

            # Verificar si hay colisión con el borde
            if self.track.mask.get_at((position[0], position[1])) == 0:
                break

        return distance

    def _calculate_reward(self):
        """
        Calcula la recompensa basada en el estado actual del coche.
        """
        if self.track.check_collision(self.car.mask, self.car.rect):
            return -1, True  # Penalización por colisión

        # Recompensa positiva por mantenerse en la pista y avanzar
        reward = self.car.speed / Config.MAX_SPEED
        return reward, False  # Continuar el episodio
