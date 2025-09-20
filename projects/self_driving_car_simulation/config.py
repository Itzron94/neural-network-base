# config.py

class Config:
    # Ventana
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600
    FPS = 60

    # Rutas de archivos
    ASSETS_PATH = "./assets/"
    TRACK_IMAGE = "track.png"
    CAR_IMAGE = "motorcycle.png"

    # Configuración de la pista
    START_POSITION = (54, 212)  # Coordenadas de la línea de partida
    START_ANGLE = 180  # Ángulo inicial del coche (apuntando hacia arriba)

    # Configuración del coche
    MAX_SPEED = 200.0  # Velocidad máxima en píxeles por segundo
    ACCELERATION = 100.0  # Aceleración en píxeles por segundo cuadrado
    DECELERATION = 100.0  # Desaceleración en píxeles por segundo cuadrado
    STEERING_SPEED = 120.0  # Velocidad de giro en grados por segundo
    CAR_SCALE = 0.5  # Escala del coche
