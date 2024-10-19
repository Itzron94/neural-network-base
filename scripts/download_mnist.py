import tensorflow as tf


def download_mnist():
    tf.keras.datasets.mnist.load_data()
    print("Conjunto de datos MNIST descargado exitosamente.")


if __name__ == "__main__":
    download_mnist()
