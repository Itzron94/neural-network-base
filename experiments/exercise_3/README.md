# Exercise 3: Multilayer Perceptron (MLP)

This exercise demonstrates the capabilities of Multilayer Perceptrons (MLPs) to solve non-linearly separable problems that simple perceptrons cannot handle.

## Problems Solved

### 1. XOR (Exclusive OR)
- **Type**: Classic non-linearly separable logical function
- **Input**: 2 features (bipolar: -1, 1)
- **Output**: Binary classification (0 or 1)
- **Key Challenge**: Requires hidden layer with non-linear activation to solve

### 2. Parity Discrimination
- **Type**: Digit classification by parity (odd vs even)
- **Input**: 35 features (7×5 binary digit patterns)
- **Output**: Binary classification (0=even, 1=odd)
- **Dataset**: `resources/datasets/TP3-ej3-digitos.txt` (digits 0-9)
- **Key Challenge**: Visual pattern recognition + binary classification

## Running the Exercise

### Single Experiment Mode (Recommended)

Run individual experiments using their respective configuration files:

```bash
# Run XOR experiment
  python experiments/exercise_3/train_tp3.py xor_config.yaml

# Run Parity Discrimination experiment
  python experiments/exercise_3/train_tp3.py parity_config.yaml
```

You can also specify the full path to a config file:

```bash
  python experiments/exercise_3/train_tp3.py experiments/exercise_3/xor_config.yaml
```

### Legacy Mode

Run both experiments sequentially without arguments:

```bash
# Run both XOR and Parity Discrimination
  python experiments/exercise_3/train_tp3.py
```
## Training Process
1. Forward pass: Calculate outputs through all layers
2. Calculate loss: MSE between predictions and targets
3. Backward pass: Compute gradients via backpropagation
4. Update weights: Apply optimizer (SGD + momentum)
5. Repeat until convergence

## Key Learnings

### 1. MLP vs Simple Perceptron
- **Simple Perceptron**: Can only solve linearly separable problems (AND, OR, NOT)
- **MLP**: Can solve non-linearly separable problems (XOR, parity, complex patterns)

### 2. Non-Linear Activation Functions
- Sigmoid activation in hidden layers enables learning of non-linear decision boundaries
- Critical for solving XOR and other non-trivial classification tasks

### 3. Architecture Depth
- **Shallow networks** (2→4→1): Sufficient for simple non-linear problems like XOR
- **Deeper networks** (35→20→10→1): Better for complex pattern recognition tasks

### 4. Backpropagation
- Manual implementation of backpropagation algorithm
- Gradient calculation through multiple layers
- Weight updates via SGD with momentum optimizer



