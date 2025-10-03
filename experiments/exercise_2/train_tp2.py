import numpy as np
from pandas import read_csv, DataFrame
import sys
from pathlib import Path
import matplotlib.pyplot as plt
# Agregar el directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from neural_network.core.perceptron import Perceptron

DATASET_PATH = "./resources/datasets/TP3-ej2-conjunto.csv"


def train_perceptron(df: DataFrame, perceptron: Perceptron, learning_rate: float = 0.1, max_epochs: int = 100, seed: int = 42):
    """
    Entrenar un perceptrón simple para el problema especificado.
    
    Args:
        problem_type: "and" o "xor"
        learning_rate: Tasa de aprendizaje
        max_epochs: Número máximo de épocas
        seed: Semilla para reproducibilidad
    """
    print(f"\n{'='*50}")
    print(f"ENTRENANDO PERCEPTRÓN")
    print(f"{'='*50}")
    
    # Configurar semilla
    if seed is not None:
        np.random.seed(seed)
    
    # Crear datos según el problema
    y = df['y'].values
    X = df.drop(columns=['y']).values
    
    print(f"\nDatos de entrenamiento:")
    print("Entrada  | Salida")
    print("-" * 17)
    
    # Crear perceptrón usando la clase existente
    print(f"\nParámetros:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Seed: {seed}")
    print(f"  - Pesos iniciales: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}]")
    print(f"  - Bias inicial: {perceptron.weights[-1]:.4f}")
    
    # Algoritmo de entrenamiento clásico del perceptrón
    print(f"\n--- INICIANDO ENTRENAMIENTO ---")
    converged = False
    
    for epoch in range(max_epochs):
        total_errors = 0
        epoch_updates = 0
        
        # Entrenar con cada muestra individualmente
        for i, (inputs, target) in enumerate(zip(X, y)):
            # Preparar entrada como matriz (1 sample)
            x_input = inputs.reshape(1, -1)
            
            # Forward pass
            prediction = perceptron.calculate_output(x_input)[0]
            
            # Calcular error
            error = target - prediction
            
            if error != 0:
                total_errors += abs(error)
                epoch_updates += 1
                
                # Actualización manual de pesos (algoritmo clásico de perceptrón)
                # w = w + lr * error * x
                perceptron.weights[:-1] += learning_rate * error * inputs
                # b = b + lr * error
                perceptron.weights[-1] += learning_rate * error
                
                # print(f"  Época {epoch+1:2d}, Muestra {i+1}: Error={error:2}, "
                #       f"Pesos=[{perceptron.weights[0]:6.3f}, {perceptron.weights[1]:6.3f}, {perceptron.weights[2]:6.3f}], "
                #       f"Bias={perceptron.weights[-1]:6.3f}")
        
        # Calcular accuracy de la época
        predictions = []
        for inputs in X:
            x_input = inputs.reshape(1, -1)
            pred = perceptron.calculate_output(x_input)[0]
            predictions.append(pred)
        
        l1_error = np.mean(np.abs(np.array(predictions) - y))
        print(f"Época {epoch+1:2d}: Errores={total_errors}, L1 error={l1_error:.3f}, Updates={epoch_updates}")
        
        # Verificar convergencia
        if l1_error < 1.0:
            converged = True
            print(f"\n¡Convergencia alcanzada en época {epoch + 1}!")
            break
    
    if not converged:
        print(f"\nNo convergió en {max_epochs} épocas")
    
    # Evaluación final
    print(f"\n--- EVALUACIÓN FINAL ---")
    print("Entrada   | Esperado | Predicho")
    print("-" * 40)
    
    final_predictions = []
    for i, (inputs, target) in enumerate(zip(X, y)):
        x_input = inputs.reshape(1, -1)
        pred = perceptron.calculate_output(x_input)[0]
        final_predictions.append(pred)
        print(f"[{inputs[0]}, {inputs[1]}, {inputs[2]}] |    {target}    |    {pred}")
    
    final_l1 = np.mean(np.abs(np.array(final_predictions)-y))
    print(f"\nL1: {final_l1:.3f} ({final_l1*100:.1f}%)")
    print(f"Pesos finales: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}, {perceptron.weights[2]:.4f}]")
    print(f"Bias final: {perceptron.weights[-1]:.4f}")
    
    return final_l1, converged, perceptron


if __name__ == "__main__":
    print("TP3 - EJERCICIO 2: PERCEPTRÓN SIMPLE")
    print("-" * 50)
    
    #Load dataset
    df = read_csv(DATASET_PATH)
    print(f"Dataset cargado desde '{DATASET_PATH}'")
    print(f"Primeras filas del dataset:\n{df.head()}")
    print(df.info())
    #Visualize dataset in 3d surface
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # scatter = ax.scatter(df['x1'], df['x2'], df['x3'], 
    #                     c=df['y'], cmap='viridis', 
    #                     s=50, alpha=0.8)

    # plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20, label='y values')

    # ax.set_xlabel('X1')
    # ax.set_ylabel('X2')
    # ax.set_zlabel('X3')
    # ax.set_title('3D Scatter Plot')

    # plt.show()
    # Entrenar y evaluar funcionamiento con perceptron lineal
    l_mdl = Perceptron(num_inputs=3, activation_type="LINEAR")
    l1, cvrgd, trained_mdl = train_perceptron(df, l_mdl, learning_rate=0.001, max_epochs=100, seed=42)
    print(f'Final L1 loss (LINEAR): {l1:.3f}, Convergió: {cvrgd}')
    # train_tp2()