# Neural Network Base

Una implementación base de una red neuronal multicapa (MLP) en Python desde cero, diseñada para ser fácilmente extensible para diversos proyectos de reconocimiento y clasificación.

## Características

- Arquitectura modular con clases para Perceptron, Layer y NeuralNetwork.
- Soporte para múltiples funciones de activación: ReLU, Sigmoid, Softmax.
- Implementación de la función de pérdida de entropía cruzada.
- Entrenamiento por lotes con Descenso de Gradiente Estocástico (SGD).
- Funcionalidad para guardar y cargar pesos de la red.
- Ejemplos de uso con el conjunto de datos MNIST y operaciones lógicas básicas.
- Pruebas unitarias para asegurar la integridad del código.

## Instalación

1. Clona el repositorio:
    ```bash
    git clone https://github.com/FelipeCupito/neural-network-base.git
    ```
2. Navega al directorio del proyecto:
    ```bash
    cd neural-network-base
    ```
3. (Opcional) Crea un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```
4. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```

## Uso

### Entrenar con MNIST

Ejecuta el script de ejemplo para entrenar la red neuronal con el conjunto de datos MNIST:

```bash
python examples/mnist_example.py
