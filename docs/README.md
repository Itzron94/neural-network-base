# Neural Network Base - Documentation

Bienvenido a la documentación del proyecto **Neural Network Base**, una implementación desde cero de una red neuronal multicapa (MLP) en Python para el reconocimiento de dígitos manuscritos utilizando el conjunto de datos **MNIST**. Esta documentación está diseñada para proporcionar una comprensión profunda de cada componente del código fuente ubicado en la carpeta `src/`, facilitando a cualquier persona entender cómo funciona cada parte y por qué se implementa de esa manera.

## **Contenido**

1. [Estructura del Proyecto](#estructura-del-proyecto)
2. [Descripción de los Módulos](#descripción-de-los-módulos)
    - [1. `activations.py`](#1-activationspy)
    - [2. `loss_functions.py`](#2-loss_functionspy)
    - [3. `perceptron.py`](#3-perceptronpy)
    - [4. `layer.py`](#4-layerpy)
    - [5. `neural_network.py`](#5-neural_networkpy)
    - [6. `utils.py`](#6-utilspy)
3. [Interconexión de Componentes](#interconexión-de-componentes)
4. [Funcionamiento General](#funcionamiento-general)
5. [Consideraciones Técnicas](#consideraciones-técnicas)
6. [Conclusión](#conclusión)

---

## **Estructura del Proyecto**

Antes de sumergirnos en la descripción detallada de cada módulo, es útil comprender cómo está organizado el código fuente dentro de la carpeta `src/`.

```
src/
├── __init__.py
├── activations.py
├── loss_functions.py
├── perceptron.py
├── layer.py
├── neural_network.py
└── utils.py
```

- **`__init__.py`**: Archivo que inicializa el paquete `src` y expone las clases y enumeraciones principales para facilitar su importación en otros módulos.
- **`activations.py`**: Define las funciones de activación utilizadas en la red neuronal.
- **`loss_functions.py`**: Contiene las funciones de pérdida que se utilizan durante el entrenamiento.
- **`perceptron.py`**: Define la clase `Perceptron`, que representa una única neurona en la red.
- **`layer.py`**: Define la clase `Layer`, que representa una capa completa de perceptrones.
- **`neural_network.py`**: Define la clase `NeuralNetwork`, que orquesta todas las capas para formar la red neuronal completa.
- **`utils.py`**: Contiene funciones utilitarias, como la carga y preprocesamiento del conjunto de datos MNIST.

---

## **Descripción de los Módulos**

A continuación, se ofrece una explicación detallada de cada uno de los módulos dentro de `src/`.

### **1. `activations.py`**

Este módulo se encarga de definir las funciones de activación que se utilizarán en los perceptrones de la red neuronal. Las funciones de activación son esenciales para introducir no linealidades en el modelo, permitiendo que la red aprenda representaciones complejas de los datos.

#### **Componentes Principales**

- **Enumeración `ActivationFunctionType`**: Define los tipos de funciones de activación disponibles.
  
  ```python
  class ActivationFunctionType(Enum):
      STEP = auto()
      SIGMOID = auto()
      RELU = auto()
      SOFTMAX = auto()
  ```

  - **`STEP`**: Función escalón, que devuelve 1 si la entrada es mayor o igual a 0, y 0 en caso contrario.
  - **`SIGMOID`**: Función sigmoide, que mapea cualquier valor real a un rango entre 0 y 1.
  - **`RELU`**: Unidad Lineal Rectificada (ReLU), que devuelve 0 si la entrada es negativa y el valor de la entrada si es positiva.
  - **`SOFTMAX`**: Función softmax, utilizada principalmente en la capa de salida para clasificación multiclase.

- **Clase Abstracta `ActivationFunction`**: Define la interfaz para todas las funciones de activación, garantizando que cada implementación tenga métodos para activar y calcular la derivada.

  ```python
  class ActivationFunction(ABC):
      @abstractmethod
      def activate(self, x: np.ndarray) -> np.ndarray:
          """Aplica la función de activación."""
          pass
  
      @abstractmethod
      def derivative(self, x: float) -> float:
          """Calcula la derivada de la función de activación."""
          pass
  ```

- **Clases Concretas de Funciones de Activación**:

  - **`StepActivation`**:
  
    ```python
    class StepActivation(ActivationFunction):
        def activate(self, x: np.ndarray) -> np.ndarray:
            return np.where(x >= 0, 1.0, 0.0)
    
        def derivative(self, x: float) -> float:
            # La derivada de la función escalón es 0 en todas partes excepto en discontinuidades
            return 0.0
    ```
    
    - **`activate`**: Implementa la función escalón.
    - **`derivative`**: La derivada es cero en todas partes excepto en los puntos de discontinuidad, donde no está definida. Por simplicidad, se retorna 0.0.

  - **`SigmoidActivation`**:
  
    ```python
    class SigmoidActivation(ActivationFunction):
        def activate(self, x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x))
    
        def derivative(self, x: float) -> float:
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
    ```
    
    - **`activate`**: Implementa la función sigmoide.
    - **`derivative`**: Calcula la derivada de la función sigmoide, que es sigmoide(x) * (1 - sigmoide(x)).

  - **`ReLUActivation`**:
  
    ```python
    class ReLUActivation(ActivationFunction):
        def activate(self, x: np.ndarray) -> np.ndarray:
            return np.maximum(0.0, x)
    
        def derivative(self, x: float) -> float:
            return 1.0 if x > 0 else 0.0
    ```
    
    - **`activate`**: Implementa la función ReLU.
    - **`derivative`**: Retorna 1.0 si x es mayor que 0, de lo contrario 0.0.

  - **`SoftmaxActivation`**:
  
    ```python
    class SoftmaxActivation(ActivationFunction):
        def activate(self, x: np.ndarray) -> np.ndarray:
            e_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Estabilización numérica
            return e_x / e_x.sum(axis=0, keepdims=True)
        
        def derivative(self, x: float) -> float:
            # La derivada de Softmax se maneja junto con la entropía cruzada
            return 1.0  # Placeholder, no se utiliza directamente
    ```
    
    - **`activate`**: Implementa la función softmax, que convierte un vector de valores en un vector de probabilidades.
    - **`derivative`**: La derivada de softmax es compleja y generalmente se maneja junto con la entropía cruzada, por lo que se retorna un valor placeholder (1.0).

- **Función de Factoría `get_activation_function`**: Devuelve una instancia de la función de activación correspondiente según el tipo especificado.

  ```python
  def get_activation_function(act_type: ActivationFunctionType) -> ActivationFunction:
      if act_type == ActivationFunctionType.STEP:
          return StepActivation()
      elif act_type == ActivationFunctionType.SIGMOID:
          return SigmoidActivation()
      elif act_type == ActivationFunctionType.RELU:
          return ReLUActivation()
      elif act_type == ActivationFunctionType.SOFTMAX:
          return SoftmaxActivation()
      else:
          raise ValueError(f"Tipo de función de activación '{act_type}' no soportada.")
  ```

  - **Propósito**: Facilitar la selección y creación de la función de activación deseada, evitando condicionales repetitivos en otras partes del código.

#### **Resumen**

El módulo `activations.py` es crucial para introducir no linealidades en la red neuronal. Cada clase de función de activación implementa métodos para activar entradas y calcular derivadas, fundamentales para el proceso de retropropagación durante el entrenamiento. La factoría `get_activation_function` simplifica la creación de instancias de estas funciones, promoviendo un diseño limpio y modular.

---

### **2. `loss_functions.py`**

Este módulo define las funciones de pérdida utilizadas para cuantificar la discrepancia entre las predicciones de la red y las etiquetas reales durante el entrenamiento. La función de pérdida guía el ajuste de los pesos de la red para mejorar su rendimiento.

#### **Componentes Principales**

- **Función `cross_entropy_loss`**: Implementa la pérdida de entropía cruzada, una medida comúnmente utilizada en problemas de clasificación multiclase.

  ```python
  def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
      """
      Calcula la pérdida de entropía cruzada.
      
      :param y_true: Array de etiquetas verdaderas (one-hot).
      :param y_pred: Array de predicciones (probabilidades).
      :return: Pérdida de entropía cruzada.
      """
      # Añadir un pequeño valor para evitar log(0)
      epsilon = 1e-12
      y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
      return -np.sum(y_true * np.log(y_pred))
  ```

  - **`y_true`**: Etiquetas verdaderas codificadas en formato one-hot. Por ejemplo, la etiqueta 3 se representa como `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`.
  - **`y_pred`**: Predicciones de la red, representadas como probabilidades sumando 1 para cada muestra.
  - **`epsilon`**: Un valor pequeño para evitar el cálculo del logaritmo de cero, lo que causaría errores numéricos.
  - **`np.clip`**: Restringe los valores de `y_pred` para que estén dentro del rango `[epsilon, 1 - epsilon]`.
  - **Cálculo de la pérdida**: La entropía cruzada se calcula sumando el producto de las etiquetas verdaderas y el logaritmo de las predicciones.

#### **Resumen**

La entropía cruzada es ideal para tareas de clasificación multiclase porque penaliza fuertemente las predicciones incorrectas. Implementarla correctamente es esencial para que el algoritmo de entrenamiento ajuste los pesos de manera efectiva.

---

### **3. `perceptron.py`**

Este módulo define la clase `Perceptron`, que representa una única neurona en la red neuronal. Cada perceptrón tiene sus propios pesos y bias, y utiliza una función de activación específica para transformar las entradas ponderadas en una salida.

#### **Componentes Principales**

- **Clase `Perceptron`**: Define la estructura y comportamiento de una neurona.

  ```python
  class Perceptron:
      def __init__(self, num_inputs: int, activation_type: ActivationFunctionType = ActivationFunctionType.SIGMOID) -> None:
          """
          Inicializa un perceptrón con pesos aleatorios y bias.
  
          :param num_inputs: Número de entradas que recibe el perceptrón.
          :param activation_type: Tipo de función de activación.
          """
          self.weights: np.ndarray = np.random.randn(num_inputs)
          self.bias: float = np.random.randn()
          self.activation: ActivationFunction = get_activation_function(activation_type)
          self.last_input: np.ndarray = np.array([])
          self.last_total: float = 0.0
          self.last_output: float = 0.0
  
      def predict(self, inputs: np.ndarray) -> float:
          """
          Realiza la predicción para un conjunto de entradas.
  
          :param inputs: Array de entradas.
          :return: Salida del perceptrón.
          """
          total = np.dot(self.weights, inputs) + self.bias
          return self.activation.activate(np.array([total]))[0]
  
      def calculate_output(self, inputs: np.ndarray) -> float:
          """
          Calcula la salida y almacena la suma ponderada para usar en la derivada.
  
          :param inputs: Array de entradas.
          :return: Salida del perceptrón.
          """
          self.last_input = inputs
          self.last_total = np.dot(self.weights, inputs) + self.bias
          self.last_output = self.activation.activate(np.array([self.last_total]))[0]
          return self.last_output
  
      def update_weights(self, delta: float, learning_rate: float) -> None:
          """
          Actualiza los pesos y el bias del perceptrón.
  
          :param delta: Delta calculado durante la retropropagación.
          :param learning_rate: Tasa de aprendizaje.
          """
          self.weights -= learning_rate * delta * self.last_input
          self.bias -= learning_rate * delta
  
      def get_activation_derivative(self) -> float:
          """
          Obtiene la derivada de la función de activación evaluada en la última suma ponderada.
  
          :return: Derivada de la función de activación.
          """
          return self.activation.derivative(self.last_total)
  ```

#### **Detalles de la Implementación**

- **Inicialización (`__init__`)**:
  
  - **Pesos (`self.weights`)**: Inicializados aleatoriamente utilizando una distribución normal estándar (`np.random.randn(num_inputs)`).
  - **Bias (`self.bias`)**: Inicializado aleatoriamente de manera similar a los pesos.
  - **Función de Activación (`self.activation`)**: Se obtiene mediante la factoría `get_activation_function` según el tipo especificado.
  - **Variables para el Entrenamiento**:
    - **`self.last_input`**: Almacena las últimas entradas utilizadas en la propagación hacia adelante. Útil para la actualización de pesos.
    - **`self.last_total`**: Almacena la suma ponderada (producto punto de pesos y entradas más bias).
    - **`self.last_output`**: Almacena la última salida generada por el perceptrón.

- **Método `predict`**:
  
  - Realiza una predicción sin almacenar información adicional para el entrenamiento.
  - Calcula la suma ponderada y aplica la función de activación.
  - Retorna la salida resultante.

- **Método `calculate_output`**:
  
  - Similar a `predict`, pero almacena las variables `last_input`, `last_total` y `last_output` necesarias para la retropropagación.
  - Esencial para el entrenamiento de la red neuronal.

- **Método `update_weights`**:
  
  - Actualiza los pesos y bias del perceptrón utilizando el delta calculado durante la retropropagación y la tasa de aprendizaje.
  - La actualización se realiza de manera que minimiza la función de pérdida.

- **Método `get_activation_derivative`**:
  
  - Retorna la derivada de la función de activación evaluada en la última suma ponderada (`last_total`).
  - Fundamental para calcular los deltas durante la retropropagación.

#### **Resumen**

La clase `Perceptron` encapsula la lógica de una neurona individual, manejando la propagación hacia adelante, el almacenamiento de estados para la retropropagación y la actualización de pesos y bias. Su diseño modular facilita la construcción de capas y, en última instancia, de la red neuronal completa.

---

### **4. `layer.py`**

Este módulo define la clase `Layer`, que representa una capa completa de perceptrones en la red neuronal. Una capa puede tener múltiples perceptrones, y cada uno procesa la misma entrada para generar diferentes salidas.

#### **Componentes Principales**

- **Clase `Layer`**: Define la estructura y comportamiento de una capa de perceptrones.

  ```python
  class Layer:
      def __init__(self, num_perceptrons: int, num_inputs_per_perceptron: int, activation_type: ActivationFunctionType = ActivationFunctionType.SIGMOID) -> None:
          """
          Inicializa una capa con múltiples perceptrones.
  
          :param num_perceptrons: Número de perceptrones en la capa.
          :param num_inputs_per_perceptron: Número de entradas por perceptrón.
          :param activation_type: Tipo de función de activación para los perceptrones.
          """
          self.perceptrons: List[Perceptron] = [
              Perceptron(num_inputs_per_perceptron, activation_type) for _ in range(num_perceptrons)
          ]
          self.outputs: np.ndarray = np.array([])
  
      def forward(self, inputs: np.ndarray) -> np.ndarray:
          """
          Propagación hacia adelante para la capa.
  
          :param inputs: Array de entradas.
          :return: Array de salidas de la capa.
          """
          self.outputs = np.array([perceptron.calculate_output(inputs) for perceptron in self.perceptrons])
          return self.outputs
  ```

#### **Detalles de la Implementación**

- **Inicialización (`__init__`)**:
  
  - **Perceptrones (`self.perceptrons`)**: Se crea una lista de instancias de `Perceptron`, cada una con el número especificado de entradas y el tipo de función de activación.
  - **Salidas (`self.outputs`)**: Inicialmente vacío, almacenará las salidas de todos los perceptrones después de la propagación hacia adelante.

- **Método `forward`**:
  
  - Realiza la propagación hacia adelante para todos los perceptrones de la capa.
  - Itera sobre cada perceptrón, calcula su salida utilizando las entradas proporcionadas y almacena todas las salidas en `self.outputs`.
  - Retorna las salidas de la capa como un array de NumPy.

#### **Resumen**

La clase `Layer` gestiona un grupo de perceptrones, facilitando la propagación hacia adelante de las entradas a través de la capa. Su diseño modular permite construir fácilmente redes neuronales con múltiples capas, cada una con diferentes tamaños y funciones de activación según sea necesario.

---

### **5. `neural_network.py`**

Este módulo define la clase `NeuralNetwork`, que representa la red neuronal completa. Orquesta las capas, gestiona el proceso de entrenamiento, evalúa el rendimiento y proporciona funcionalidades para guardar y cargar los pesos de la red.

#### **Componentes Principales**

- **Clase `NeuralNetwork`**: Define la estructura y comportamiento de la red neuronal completa.

  ```python
  class NeuralNetwork:
      def __init__(
          self, 
          topology: List[int], 
          activation_type: ActivationFunctionType = ActivationFunctionType.SIGMOID, 
          learning_rate: float = 0.01, 
          epochs: int = 1000
      ) -> None:
          """
          Inicializa la red neuronal con una topología específica.
  
          :param topology: Lista que define el número de perceptrones en cada capa.
                           Por ejemplo, [784, 128, 64, 10] representa:
                           - 784 perceptrones en la capa de entrada
                           - 128 en la primera capa oculta
                           - 64 en la segunda capa oculta
                           - 10 en la capa de salida
          :param activation_type: Tipo de función de activación para las capas ocultas.
          :param learning_rate: Tasa de aprendizaje para la actualización de pesos.
          :param epochs: Número de iteraciones sobre el conjunto de entrenamiento.
          """
          if len(topology) < 2:
              raise ValueError("La topología debe tener al menos dos capas (entrada y salida).")
          
          self.layers: List[Layer] = []
          self.learning_rate: float = learning_rate
          self.epochs: int = epochs
  
          # Crear capas ocultas y de salida
          for i in range(1, len(topology)):
              # La capa de salida usa Softmax
              activation = ActivationFunctionType.SOFTMAX if i == len(topology) -1 else activation_type
              layer = Layer(
                  num_perceptrons=topology[i],
                  num_inputs_per_perceptron=topology[i-1],
                  activation_type=activation
              )
              self.layers.append(layer)
  
      def forward(self, inputs: np.ndarray) -> np.ndarray:
          """
          Realiza la propagación hacia adelante a través de todas las capas.
  
          :param inputs: Array de entradas.
          :return: Salida final de la red.
          """
          for layer in self.layers:
              inputs = layer.forward(inputs)
          return inputs
  
      def train(self, training_inputs: np.ndarray, training_labels: np.ndarray, batch_size: int = 32) -> None:
          """
          Entrena la red neuronal utilizando el algoritmo de retropropagación con Softmax y Entropía Cruzada.
          Implementa entrenamiento por lotes.
  
          :param training_inputs: Array de entradas de entrenamiento.
          :param training_labels: Array de etiquetas de entrenamiento (one-hot).
          :param batch_size: Tamaño del lote para entrenamiento.
          """
          num_samples = training_inputs.shape[0]
          for epoch in range(1, self.epochs + 1):
              total_loss = 0.0
              # Mezclar los datos para cada época
              indices = np.arange(num_samples)
              np.random.shuffle(indices)
              shuffled_inputs = training_inputs[indices]
              shuffled_labels = training_labels[indices]
              
              for start_idx in range(0, num_samples, batch_size):
                  end_idx = min(start_idx + batch_size, num_samples)
                  batch_inputs = shuffled_inputs[start_idx:end_idx]
                  batch_labels = shuffled_labels[start_idx:end_idx]
                  
                  # Propagación hacia adelante
                  batch_outputs = self.forward(batch_inputs)
  
                  # Calcular pérdida de entropía cruzada
                  batch_loss = cross_entropy_loss(batch_labels, batch_outputs)
                  total_loss += batch_loss
  
                  # Retropropagación del error
                  deltas: List[np.ndarray] = []
                  
                  # Delta para la capa de salida (simplificado para Softmax + Entropía Cruzada)
                  delta_output = batch_outputs - batch_labels  # derivada simplificada
                  deltas.insert(0, delta_output)
  
                  # Calcular delta para las capas ocultas
                  for l in range(len(self.layers) - 2, -1, -1):
                      current_layer = self.layers[l]
                      next_layer = self.layers[l + 1]
                      delta = np.zeros((len(current_layer.perceptrons), batch_size))
                      for i, perceptron in enumerate(current_layer.perceptrons):
                          # Sumar los deltas ponderados de la siguiente capa
                          weights = np.array([next_layer.perceptrons[j].weights[i] for j in range(len(next_layer.perceptrons))])
                          sum_delta = np.dot(weights, deltas[0])
                          delta[i] = sum_delta * current_layer.perceptrons[i].get_activation_derivative()
                      deltas.insert(0, delta)
  
                  # Actualizar pesos y biases
                  for l, layer in enumerate(self.layers):
                      # Usar las entradas del lote para la primera capa, o las salidas de la capa anterior
                      inputs_to_use = batch_inputs if l == 0 else self.layers[l - 1].outputs
                      for j, perceptron in enumerate(layer.perceptrons):
                          # Promediar los deltas sobre el lote
                          perceptron.weights -= self.learning_rate * np.mean(deltas[l][j] * inputs_to_use, axis=0)
                          perceptron.bias -= self.learning_rate * np.mean(deltas[l][j])
  
              # Opcional: Imprimir la pérdida cada cierta cantidad de épocas
              if epoch % 100 == 0 or epoch == 1:
                  print(f"Época {epoch}/{self.epochs} - Pérdida: {total_loss}")
  
      def evaluate(self, test_inputs: np.ndarray, test_labels: np.ndarray) -> float:
          """
          Evalúa la precisión de la red neuronal en un conjunto de datos de prueba.
  
          :param test_inputs: Array de entradas de prueba.
          :param test_labels: Array de etiquetas de prueba (one-hot).
          :return: Precisión de la red.
          """
          predictions = self.forward(test_inputs)
          predicted_classes = np.argmax(predictions, axis=1)
          true_classes = np.argmax(test_labels, axis=1)
          accuracy = np.mean(predicted_classes == true_classes)
          return accuracy
  
      def predict(self, inputs: np.ndarray) -> np.ndarray:
          """
          Realiza una predicción con la red neuronal.
  
          :param inputs: Array de entradas.
          :return: Array de salidas de la red.
          """
          return self.forward(inputs)
  
      def save_weights(self, file_path: str) -> None:
          """
          Guarda los pesos y biases de la red neuronal en un archivo.
  
          :param file_path: Ruta del archivo donde se guardarán los pesos.
          """
          data = {}
          for layer_num, layer in enumerate(self.layers):
              # Convertir los pesos a listas para evitar problemas de broadcasting
              layer_weights = [perceptron.weights.tolist() for perceptron in layer.perceptrons]
              layer_biases = [perceptron.bias for perceptron in layer.perceptrons]
              data[f'layer_{layer_num}_weights'] = layer_weights
              data[f'layer_{layer_num}_biases'] = layer_biases
          # Guardar como un archivo .npz usando argumentos de palabra clave
          np.savez(file_path, **data)
          print(f"Pesos guardados en '{file_path}'.")
  
      def load_weights(self, file_path: str) -> None:
          """
          Carga los pesos y biases de la red neuronal desde un archivo.
  
          :param file_path: Ruta del archivo desde donde se cargarán los pesos.
          """
          if not os.path.exists(file_path):
              raise FileNotFoundError(f"El archivo '{file_path}' no existe.")
          
          data = np.load(file_path, allow_pickle=True)
          num_layers = len(self.layers)
          for layer_num in range(num_layers):
              layer_weights = data[f'layer_{layer_num}_weights']
              layer_biases = data[f'layer_{layer_num}_biases']
              for perceptron, w, b in zip(self.layers[layer_num].perceptrons, layer_weights, layer_biases):
                  perceptron.weights = np.array(w)
                  perceptron.bias = float(b)
          print(f"Pesos cargados desde '{file_path}'.")
  ```

#### **Detalles de la Implementación**

- **Inicialización (`__init__`)**:
  
  - **`topology`**: Lista que define la estructura de la red neuronal. Por ejemplo, `[784, 128, 64, 10]` significa:
    - 784 perceptrones en la capa de entrada (equivalente a 28x28 píxeles de MNIST).
    - 128 perceptrones en la primera capa oculta.
    - 64 perceptrones en la segunda capa oculta.
    - 10 perceptrones en la capa de salida (uno por cada dígito del 0 al 9).
  - **`activation_type`**: Tipo de función de activación para las capas ocultas. Por defecto, `SIGMOID`, pero generalmente se utiliza `RELU` para capas ocultas en tareas modernas.
  - **`learning_rate`**: Tasa de aprendizaje para el ajuste de pesos durante el entrenamiento.
  - **`epochs`**: Número de épocas (iteraciones completas sobre el conjunto de entrenamiento).
  
  - **Creación de Capas**:
    - Itera sobre la topología para crear cada capa.
    - La capa de salida utiliza la función de activación `Softmax`, adecuada para clasificación multiclase.
    - Las capas ocultas utilizan la función de activación especificada (`SIGMOID` o `RELU`).

- **Método `forward`**:
  
  - Realiza la propagación hacia adelante a través de todas las capas de la red.
  - Itera sobre cada capa, pasando las salidas de una capa como entradas a la siguiente.
  - Retorna la salida final de la red.

- **Método `train`**:
  
  - Entrena la red neuronal utilizando retropropagación y descenso de gradiente estocástico (SGD) con entrenamiento por lotes.
  - **Entrenamiento por Lote**:
    - Divide los datos de entrenamiento en lotes (`batch_size`), lo que mejora la eficiencia y estabilidad del entrenamiento.
  - **Proceso de Entrenamiento**:
    1. **Mezcla de Datos**: Aleatoriza el orden de las muestras para cada época, evitando patrones que puedan afectar negativamente el entrenamiento.
    2. **Propagación hacia Adelante**: Calcula las predicciones de la red para cada lote.
    3. **Cálculo de la Pérdida**: Utiliza la función de pérdida de entropía cruzada para medir la discrepancia entre las predicciones y las etiquetas reales.
    4. **Retropropagación**:
       - Calcula los deltas (errores) para cada capa, comenzando desde la capa de salida hacia las capas ocultas.
       - Ajusta los pesos y biases de cada perceptrón basándose en estos deltas y la tasa de aprendizaje.
  - **Impresión de la Pérdida**: Opcionalmente imprime la pérdida total cada cierto número de épocas para monitorear el progreso.

- **Método `evaluate`**:
  
  - Evalúa la precisión de la red neuronal en un conjunto de datos de prueba.
  - Calcula las predicciones, determina las clases predichas y compara con las clases verdaderas para calcular la precisión.

- **Método `predict`**:
  
  - Realiza una predicción para una o más entradas sin realizar ajustes en la red.
  - Útil para inferencia una vez que la red ha sido entrenada.

- **Métodos `save_weights` y `load_weights`**:
  
  - **`save_weights`**: Guarda los pesos y biases de todas las capas en un archivo `.npz`, almacenando cada capa por separado para evitar problemas de broadcasting.
  - **`load_weights`**: Carga los pesos y biases desde un archivo `.npz` y los asigna a los perceptrones correspondientes en cada capa.

#### **Consideraciones Importantes**

- **Retropropagación Simplificada para Softmax + Entropía Cruzada**:
  
  - Cuando se usa Softmax en la capa de salida junto con la pérdida de entropía cruzada, la derivada de la pérdida respecto a la entrada de Softmax simplifica a `output - label`.
  - Esto agiliza el cálculo de deltas en la capa de salida.

- **Actualización de Pesos**:
  
  - Los pesos y biases se actualizan restando el producto de la tasa de aprendizaje y el delta correspondiente.
  - Esto se realiza utilizando la media de los deltas sobre el lote para una actualización más estable.

#### **Resumen**

La clase `NeuralNetwork` es el núcleo del proyecto, orquestando la creación de capas, la propagación hacia adelante, el entrenamiento mediante retropropagación y la evaluación del rendimiento. Además, proporciona funcionalidades para persistir el estado de la red mediante el guardado y carga de pesos, lo que facilita la reutilización y el despliegue del modelo entrenado.

---

### **6. `utils.py`**

Este módulo contiene funciones utilitarias que facilitan tareas como la carga y preprocesamiento del conjunto de datos MNIST. Separar estas funciones en un módulo propio promueve la reutilización y mantiene el código limpio.

#### **Componentes Principales**

- **Función `load_mnist`**: Carga y preprocesa el conjunto de datos MNIST, retornando los datos de entrenamiento y prueba listos para ser utilizados por la red neuronal.

  ```python
  def load_mnist():
      """
      Carga y preprocesa el conjunto de datos MNIST.
  
      :return: Tupla de (x_train, y_train, x_test, y_test).
      """
      # Cargar el conjunto de datos MNIST
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
      
      # Aplanar las imágenes de 28x28 a 784 dimensiones
      x_train = x_train.reshape(-1, 28*28).astype(np.float32)
      x_test = x_test.reshape(-1, 28*28).astype(np.float32)
      
      # Normalizar los valores de píxeles a [0, 1]
      x_train /= 255.0
      x_test /= 255.0
      
      # Convertir etiquetas a vectores one-hot
      y_train = np.eye(10)[y_train]
      y_test = np.eye(10)[y_test]
      
      return x_train, y_train, x_test, y_test
  ```

#### **Detalles de la Implementación**

- **Carga de Datos**:
  
  - Utiliza `tf.keras.datasets.mnist.load_data()` para descargar y cargar el conjunto de datos MNIST.
  - Este método devuelve dos tuplas: una para entrenamiento y otra para prueba, cada una con imágenes y etiquetas correspondientes.

- **Preprocesamiento de Imágenes**:
  
  - **Aplanamiento**: Convierte las imágenes de formato 2D (28x28) a vectores 1D de 784 dimensiones para que sean compatibles con la entrada de la red neuronal.
  
    ```python
    x_train = x_train.reshape(-1, 28*28).astype(np.float32)
    x_test = x_test.reshape(-1, 28*28).astype(np.float32)
    ```
  
  - **Normalización**: Escala los valores de píxeles de [0, 255] a [0, 1], lo que mejora la estabilidad y eficiencia del entrenamiento.
  
    ```python
    x_train /= 255.0
    x_test /= 255.0
    ```

- **Preprocesamiento de Etiquetas**:
  
  - **Codificación One-Hot**: Convierte las etiquetas de dígitos (0-9) a vectores one-hot, donde cada etiqueta se representa como un vector con un 1 en la posición correspondiente y 0s en las demás.
  
    ```python
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    ```
  
  - **Propósito**: Facilita el cálculo de la pérdida de entropía cruzada y la retropropagación en la red neuronal.

#### **Resumen**

El módulo `utils.py` proporciona una función esencial para preparar los datos de entrada y salida de manera que sean adecuados para el entrenamiento y evaluación de la red neuronal. Este enfoque modular facilita la reutilización y adaptación de la función para diferentes conjuntos de datos o tareas en el futuro.

---

## **Interconexión de Componentes**

Para comprender cómo todos los módulos trabajan juntos para formar una red neuronal funcional, es crucial visualizar la interrelación entre ellos:

1. **`utils.py`**: Carga y preprocesa los datos de MNIST, retornando las matrices de entrada (`x_train`, `x_test`) y las etiquetas (`y_train`, `y_test`) en formato one-hot.

2. **`activations.py`**: Define las funciones de activación que transforman las salidas ponderadas de cada perceptrón, introduciendo no linealidades necesarias para el aprendizaje.

3. **`perceptron.py`**: Cada `Perceptron` utiliza una función de activación para procesar sus entradas y generar una salida. Almacena información necesaria para la retropropagación.

4. **`layer.py`**: Cada `Layer` contiene múltiples perceptrones. La propagación hacia adelante en una capa implica la activación de cada perceptrón con las mismas entradas, produciendo un vector de salidas para la capa.

5. **`neural_network.py`**: Orquesta todas las capas, gestionando la propagación hacia adelante a través de ellas, el entrenamiento mediante retropropagación, y la evaluación del rendimiento del modelo.

6. **`loss_functions.py`**: Proporciona la función de pérdida que cuantifica la diferencia entre las predicciones de la red y las etiquetas reales, esencial para guiar el entrenamiento.

7. **`__init__.py`**: Facilita la importación de las clases y enumeraciones principales, permitiendo un acceso sencillo a través del paquete `src`.

---

## **Funcionamiento General**

A continuación, se detalla el flujo general desde la carga de datos hasta la predicción:

1. **Carga y Preprocesamiento de Datos**:
   
   - Se utiliza la función `load_mnist` de `utils.py` para obtener y preparar los datos de MNIST.
   - Las imágenes se aplanan y normalizan, y las etiquetas se convierten a formato one-hot.

2. **Inicialización de la Red Neuronal**:
   
   - Se crea una instancia de `NeuralNetwork` definiendo la topología, el tipo de función de activación para las capas ocultas, la tasa de aprendizaje y el número de épocas.
   - Por ejemplo:
     
     ```python
     topology = [784, 128, 64, 10]
     nn = NeuralNetwork(
         topology=topology, 
         activation_type=ActivationFunctionType.RELU,  # ReLU para capas ocultas
         learning_rate=0.01, 
         epochs=1000
     )
     ```
   
   - Esto crea una red con:
     - **Capa de Entrada**: 784 nodos.
     - **Primera Capa Oculta**: 128 perceptrones con activación ReLU.
     - **Segunda Capa Oculta**: 64 perceptrones con activación ReLU.
     - **Capa de Salida**: 10 perceptrones con activación Softmax.

3. **Entrenamiento de la Red**:
   
   - Se llama al método `train` de la instancia `NeuralNetwork`, pasando las entradas y etiquetas de entrenamiento.
   
     ```python
     nn.train(x_train, y_train, batch_size=64)
     ```
   
   - **Proceso de Entrenamiento**:
     - **Mezcla de Datos**: Aleatoriza el orden de las muestras para cada época.
     - **Entrenamiento por Lote**: Divide los datos en lotes (`batch_size=64`) y procesa cada lote de manera independiente.
     - **Propagación hacia Adelante**: Calcula las salidas de la red para cada lote.
     - **Cálculo de la Pérdida**: Evalúa la entropía cruzada entre las predicciones y las etiquetas reales.
     - **Retropropagación**: Calcula los deltas para cada capa y actualiza los pesos y biases de los perceptrones.
     - **Monitoreo**: Imprime la pérdida total cada 100 épocas para monitorear el progreso.

4. **Evaluación de la Red**:
   
   - Se utiliza el método `evaluate` para medir la precisión de la red en el conjunto de prueba.
   
     ```python
     accuracy = nn.evaluate(x_test, y_test)
     print(f"Precisión en el conjunto de prueba: {accuracy * 100:.2f}%")
     ```

5. **Guardar y Cargar Pesos**:
   
   - **Guardar Pesos**:
     
     ```python
     nn.save_weights("mnist_mlp_weights.npz")
     ```
   
   - **Cargar Pesos**:
     
     ```python
     nn_loaded = NeuralNetwork(
         topology=topology, 
         activation_type=ActivationFunctionType.RELU, 
         learning_rate=0.01, 
         epochs=1000
     )
     nn_loaded.load_weights("mnist_mlp_weights.npz")
     ```

6. **Predicciones**:
   
   - Se pueden realizar predicciones individuales utilizando el método `predict`.
   
     ```python
     output = nn.predict(inputs)
     predicted_label = np.argmax(output)
     ```

#### **Flujo de Datos**

1. **Entrada**: Vector de 784 dimensiones (28x28 píxeles aplanados y normalizados).
2. **Capa Oculta 1**: Procesa la entrada mediante 128 perceptrones con activación ReLU, produciendo un vector de 128 salidas.
3. **Capa Oculta 2**: Toma las 128 salidas de la capa anterior, las procesa mediante 64 perceptrones con activación ReLU, produciendo un vector de 64 salidas.
4. **Capa de Salida**: Toma las 64 salidas de la capa anterior, las procesa mediante 10 perceptrones con activación Softmax, produciendo un vector de 10 probabilidades.
5. **Salida**: Vector de 10 probabilidades que suman 1, representando la probabilidad de pertenencia a cada clase (dígito del 0 al 9).

---

## **Consideraciones Técnicas**

### **1. Funciones de Activación**

Las funciones de activación introducen no linealidades en la red, permitiendo que aprenda representaciones complejas. Cada función tiene sus propias características:

- **ReLU**:
  - Ventajas:
    - Computacionalmente eficiente.
    - Mitiga el problema del desvanecimiento del gradiente.
  - Desventajas:
    - Puede sufrir de neuronas "muertas" que nunca se activan.

- **Softmax**:
  - Utilizada en la capa de salida para problemas de clasificación multiclase.
  - Transforma un vector de valores en probabilidades que suman 1.

### **2. Función de Pérdida: Entropía Cruzada**

La entropía cruzada es adecuada para problemas de clasificación multiclase porque penaliza fuertemente las predicciones incorrectas y está alineada con la función de activación Softmax, lo que simplifica la derivada durante la retropropagación.

### **3. Retropropagación**

El algoritmo de retropropagación ajusta los pesos de la red neuronal para minimizar la función de pérdida. En esta implementación:

- **Capa de Salida**: El delta se calcula como `output - label` debido a la combinación de Softmax y entropía cruzada.
- **Capas Ocultas**: Los deltas se calculan multiplicando el delta de la capa siguiente por los pesos correspondientes y la derivada de la función de activación.

### **4. Entrenamiento por Lote**

El entrenamiento por lote mejora la eficiencia y estabilidad del entrenamiento al procesar múltiples muestras simultáneamente. Además, ayuda a aprovechar las optimizaciones vectorizadas de NumPy.

### **5. Guardado y Carga de Pesos**

Guardar los pesos permite reutilizar redes entrenadas sin necesidad de reentrenarlas, lo que es útil para despliegues y análisis posteriores. La carga de pesos requiere que la topología de la red coincida exactamente con la de la red en el momento del guardado.

---

## **Conclusión**

Este proyecto proporciona una implementación modular y extensible de una red neuronal multicapa desde cero en Python. Cada componente está diseñado para ser claro y mantenible, permitiendo una fácil expansión y adaptación a diferentes tareas y conjuntos de datos. Aunque esta implementación es educativa y facilita la comprensión de los fundamentos de las redes neuronales, para aplicaciones más complejas y eficientes se recomienda utilizar bibliotecas especializadas como TensorFlow o PyTorch.

---

## **Referencias**

- [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [Neural Networks and Deep Learning - Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [An Introduction to Neural Networks - Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap1.html)

---

¡Gracias por utilizar **Neural Network Base**! Si tienes alguna pregunta o necesitas más ayuda, no dudes en contactar o contribuir al proyecto.