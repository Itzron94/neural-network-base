"""
TP1 - Ejercicio 1: Perceptrón Simple para Funciones Lógicas AND y XOR

Implementación básica del algoritmo de perceptrón simple con función escalón bipolar.
Sin usar librerías de ML, solo NumPy para operaciones matemáticas básicas.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Agregar el directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_network.core.perceptron import Perceptron


def create_and_gate_data():
    """
    Crear datos para la función lógica AND.
    Entradas: {-1,1}, {1,-1}, {-1,-1}, {1,1}
    Salidas:  {-1}, {-1}, {-1}, {1}
    """
    X = np.array([
        [-1, 1],
        [1, -1], 
        [-1, -1],
        [1, 1]
    ])
    
    y = np.array([-1, -1, -1, 1])
    
    return X, y


def create_xor_gate_data():
    """
    Crear datos para la función lógica XOR.
    Entradas: {-1,1}, {1,-1}, {-1,-1}, {1,1}
    Salidas:  {1}, {1}, {-1}, {-1}
    """
    X = np.array([
        [-1, 1],
        [1, -1],
        [-1, -1], 
        [1, 1]
    ])
    
    y = np.array([1, 1, -1, -1])
    
    return X, y


def train_perceptron(problem_type: str, learning_rate: float = 0.1, max_epochs: int = 100, seed: int = 42):
    """
    Entrenar un perceptrón simple para el problema especificado.
    
    Args:
        problem_type: "and" o "xor"
        learning_rate: Tasa de aprendizaje
        max_epochs: Número máximo de épocas
        seed: Semilla para reproducibilidad
    """
    print(f"\n{'='*50}")
    print(f"ENTRENANDO PERCEPTRÓN PARA {problem_type.upper()}")
    print(f"{'='*50}")
    
    # Configurar semilla
    if seed is not None:
        np.random.seed(seed)
    
    # Crear datos según el problema
    if problem_type == "and":
        X, y = create_and_gate_data()
        print("Problema: Función lógica AND")
    elif problem_type == "xor":
        X, y = create_xor_gate_data()
        print("Problema: Función lógica XOR")
    else:
        raise ValueError("problem_type debe ser 'and' o 'xor'")
    
    print(f"\nDatos de entrenamiento:")
    print("Entrada  | Salida")
    print("-" * 17)
    for i, (inputs, target) in enumerate(zip(X, y)):
        print(f"[{inputs[0]:2}, {inputs[1]:2}] |   {target:2}")
    
    # Crear perceptrón usando la clase existente
    perceptron = Perceptron(num_inputs=2, activation_type="STEP_BIPOLAR")
    
    print(f"\nParámetros:")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Max epochs: {max_epochs}")
    print(f"  - Seed: {seed}")
    print(f"  - Pesos iniciales: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}]")
    print(f"  - Bias inicial: {perceptron.bias:.4f}")
    
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
            prediction = perceptron.calculate_output(x_input)
            predicted_class = 1 if prediction[0] >= 0 else -1
            
            # Calcular error
            error = target - predicted_class
            
            if error != 0:
                total_errors += abs(error)
                epoch_updates += 1
                
                # Actualización manual de pesos (algoritmo clásico de perceptrón)
                # w = w + lr * error * x
                perceptron.weights += learning_rate * error * inputs
                
                # b = b + lr * error
                perceptron.bias += learning_rate * error
                
                print(f"  Época {epoch+1:2d}, Muestra {i+1}: Error={error:2}, "
                      f"Pesos=[{perceptron.weights[0]:6.3f}, {perceptron.weights[1]:6.3f}], "
                      f"Bias={perceptron.bias:6.3f}")
        
        # Calcular accuracy de la época
        predictions = []
        for inputs in X:
            x_input = inputs.reshape(1, -1)
            pred = perceptron.calculate_output(x_input)
            pred_class = 1 if pred[0] >= 0 else -1
            predictions.append(pred_class)
        
        accuracy = np.mean(np.array(predictions) == y)
        print(f"Época {epoch+1:2d}: Errores={total_errors}, Accuracy={accuracy:.3f}, Updates={epoch_updates}")
        
        # Verificar convergencia
        if total_errors == 0:
            converged = True
            print(f"\n¡Convergencia alcanzada en época {epoch + 1}!")
            break
    
    if not converged:
        print(f"\nNo convergió en {max_epochs} épocas")
    
    # Evaluación final
    print(f"\n--- EVALUACIÓN FINAL ---")
    print("Entrada   | Esperado | Predicho | ¿Correcto?")
    print("-" * 40)
    
    final_predictions = []
    for i, (inputs, target) in enumerate(zip(X, y)):
        x_input = inputs.reshape(1, -1)
        pred = perceptron.calculate_output(x_input)
        pred_class = 1 if pred[0] >= 0 else -1
        final_predictions.append(pred_class)
        
        status = "✓" if target == pred_class else "✗"
        print(f"[{inputs[0]:2}, {inputs[1]:2}] |    {target:2}    |    {pred_class:2}    |     {status}")
    
    final_accuracy = np.mean(np.array(final_predictions) == y)
    print(f"\nAccuracy: {final_accuracy:.3f} ({final_accuracy*100:.1f}%)")
    print(f"Pesos finales: [{perceptron.weights[0]:.4f}, {perceptron.weights[1]:.4f}]")
    print(f"Bias final: {perceptron.bias:.4f}")
    
    # Mostrar frontera de decisión
    if abs(perceptron.weights[1]) < 1e-10:
        print(f"\nFrontera de decisión:")
        print(f"  Línea vertical: x1 = {-perceptron.bias / perceptron.weights[0]:.4f}")
    else:
        slope = -perceptron.weights[0] / perceptron.weights[1]
        intercept = -perceptron.bias / perceptron.weights[1]
        print(f"\nFrontera de decisión:")
        print(f"  Ecuación: x2 = {slope:.4f} * x1 + {intercept:.4f}")
    
    return final_accuracy, converged, perceptron


def analyze_perceptron_limitations():
    """
    Analizar las limitaciones del perceptrón simple basado en los resultados.
    """
    print("\n" + "="*60)
    print("TP1 - ANÁLISIS DE LIMITACIONES DEL PERCEPTRÓN SIMPLE")
    print("="*60)
    
    # Entrenar ambos problemas
    and_accuracy, and_converged, and_perceptron = train_perceptron("and", max_epochs=50)
    xor_accuracy, xor_converged, xor_perceptron = train_perceptron("xor", max_epochs=50)
    
    print(f"\n{'='*60}")
    print("RESUMEN COMPARATIVO")
    print("="*60)
    
    print(f"\n🔹 Problema AND:")
    print(f"  - Accuracy final: {and_accuracy:.3f} ({and_accuracy*100:.1f}%)")
    print(f"  - ¿Converge?: {'SÍ' if and_converged else 'NO'}")
    
    print(f"\n🔹 Problema XOR:")
    print(f"  - Accuracy final: {xor_accuracy:.3f} ({xor_accuracy*100:.1f}%)")
    print(f"  - ¿Converge?: {'SÍ' if xor_converged else 'NO'}")
    
    print(f"\n{'='*60}")
    print("CONCLUSIONES")
    print("="*60)
    
    print("\n🔍 1. SEPARABILIDAD LINEAL:")
    print("   • El problema AND ES linealmente separable")
    print("   • El problema XOR NO ES linealmente separable")
    
    print("\n🚫 2. LIMITACIONES DEL PERCEPTRÓN SIMPLE:")
    print("   • Solo puede resolver problemas linealmente separables")
    print("   • Utiliza una línea recta como frontera de decisión")
    print("   • XOR requiere una frontera de decisión no lineal")
    
    print("\n🧠 3. CAPACIDAD REPRESENTACIONAL:")
    print("   • Un perceptrón simple solo puede aprender funciones lineales")
    print("   • Para problemas como XOR se necesitan redes multicapa (MLP)")
    
    print("\n📈 4. CONVERGENCIA:")
    if and_converged:
        print("   • AND converge porque es linealmente separable")
    if not xor_converged:
        print("   • XOR no converge porque no es linealmente separable")
    
    print(f"\n{'='*60}")
    print("¿QUÉ PROBLEMAS PUEDE RESOLVER EL PERCEPTRÓN SIMPLE?")
    print("="*60)
    print("✅ Funciones lógicas: AND, OR, NOT")
    print("✅ Clasificación binaria linealmente separable")
    print("✅ Regresión lineal simple")
    print("❌ Funciones lógicas: XOR, XNOR")
    print("❌ Problemas no linealmente separables")
    print("❌ Clasificación multiclase compleja")


if __name__ == "__main__":
    print("TP1 - EJERCICIO 1: PERCEPTRÓN SIMPLE")
    print("Implementación sin librerías de ML (solo NumPy)")
    print("-" * 50)
    
    # Ejecutar análisis completo
    analyze_perceptron_limitations()