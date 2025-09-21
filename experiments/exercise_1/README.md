# TP1 - Perceptrón Simple

## Descripción
Implementación del algoritmo de perceptrón simple con función de activación escalón bipolar para resolver problemas de clasificación de funciones lógicas.

## Problemas a Resolver

### Función AND
- **Entradas**: {-1,1}, {1,-1}, {-1,-1}, {1,1}
- **Salidas**: {-1}, {-1}, {-1}, {1}
- **Característica**: Linealmente separable

### Función XOR  
- **Entradas**: {-1,1}, {1,-1}, {-1,-1}, {1,1}
- **Salidas**: {1}, {1}, {-1}, {-1}
- **Característica**: NO linealmente separable

## Arquitectura del Perceptrón
- **Topología**: 2 entradas → 1 salida
- **Función de activación**: Escalón bipolar (-1/+1)
- **Algoritmo de entrenamiento**: Regla del perceptrón
- **Actualización de pesos**: w = w + η * error * x

## Ejecución
```bash
cd experiments/tp1_perceptron
python train_tp1.py
```

## Resultados Esperados

### AND Gate
- ✅ **Converge**: SÍ (problema linealmente separable)
- ✅ **Accuracy**: 100%
- ✅ **Épocas**: 10-30 épocas típicamente

### XOR Gate  
- ❌ **Converge**: NO (problema no linealmente separable)
- ❌ **Accuracy**: ~50% (aleatorio)
- ❌ **Épocas**: No converge en 100 épocas

## Análisis Teórico

### Separabilidad Lineal
Un perceptrón simple solo puede resolver problemas donde las clases pueden ser separadas por una **línea recta** (hiperplano en dimensiones superiores).

**AND es separable**: Existe una línea que separa los puntos donde AND=1 del resto.
**XOR no es separable**: No existe una línea recta que pueda separar correctamente los puntos.

### Limitaciones del Perceptrón Simple
1. **Solo funciones linealmente separables**
2. **Capacidad representacional limitada**  
3. **No puede resolver XOR sin capas ocultas**
4. **Requiere MLP para problemas no lineales**

## Archivos Generados
- `and_gate/outputs/`: Resultados del experimento AND
- `xor_gate/outputs/`: Resultados del experimento XOR
- Gráficos de convergencia y análisis
- Pesos entrenados y métricas detalladas