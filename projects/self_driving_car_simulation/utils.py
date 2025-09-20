# utils.py

import pygame


def load_image(path, filename):
    """
    Carga una imagen y maneja errores.
    """
    try:
        image = pygame.image.load(f"{path}{filename}").convert_alpha()
        return image
    except pygame.error as e:
        print(f"No se pudo cargar la imagen {filename}: {e}")
        raise SystemExit(e)


def rotate_image(image, angle):
    """
    Rota una imagen y devuelve la imagen rotada.
    """
    rotated_image = pygame.transform.rotozoom(image, -angle, 1)
    return rotated_image
