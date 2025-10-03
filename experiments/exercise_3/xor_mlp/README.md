# Exercise 3 - XOR Logical Function with Multilayer Perceptron

**Validation Exercise - NOT for presentation**

## Overview

This exercise validates the correct implementation of the multilayer perceptron by solving the XOR (exclusive OR) problem. XOR is a classic benchmark for neural networks because it is **not linearly separable**, meaning a single-layer perceptron cannot solve it. This problem requires at least one hidden layer with non-linear activation to learn the correct decision boundary.

## Problem Description

The XOR logical function outputs true (1) only when the inputs differ:

| Input 1 | Input 2 | XOR Output |
|---------|---------|------------|
| -1      | -1      | -1         |
| -1      | 1       | 1          |
| 1       | -1      | 1          |
| 1       | 1       | -1         |

### Why XOR Requires a Hidden Layer

XOR is the canonical example of a non-linearly separable problem. No single straight line can separate the positive cases ([-1,1] and [1,-1]) from the negative cases ([-1,-1] and [1,1]). This was famously demonstrated by Minsky and Papert in 1969, leading to the first "AI winter" until the discovery of backpropagation in the 1980s enabled multi-layer networks to solve such problems.

## Network Architecture

```
Input Layer (2 neurons)
    ↓
Hidden Layer (4 neurons, Sigmoid activation)
    ↓
Output Layer (1 neuron, Sigmoid activation)
```

### Architecture Details

- **Input Layer**: 2 neurons
  - Accepts bipolar inputs {-1, 1} (same format as Exercise 1)
  
- **Hidden Layer**: 4 neurons
  - Activation: Sigmoid (provides non-linearity essential for XOR)
  - Sufficient capacity to learn the XOR decision boundary
  
- **Output Layer**: 1 neuron
  - Activation: Sigmoid (outputs values between 0 and 1)
  - Binary classification: outputs are normalized to {0, 1}
    - Output < 0.5 → Class 0 (represents -1)
    - Output ≥ 0.5 → Class 1 (represents 1)

The hidden layer with sigmoid activation creates a non-linear transformation space where XOR becomes linearly separable. The 4 hidden neurons provide enough capacity to learn the necessary feature combinations. 

## Implementation Details

### Data Representation

**Input**: Bipolar format {-1, 1}
```python
X = [[-1, -1],
     [-1,  1],
     [ 1, -1],
     [ 1,  1]]
```

**Output**: Normalized to {0, 1} for sigmoid
```python
y = [[0],  # -1 XOR -1 = -1 → 0
     [1],  # -1 XOR  1 =  1 → 1
     [1],  #  1 XOR -1 =  1 → 1
     [0]]  #  1 XOR  1 = -1 → 0
```

The outputs are normalized to match the sigmoid activation range (0 to 1), then converted back to bipolar {-1, 1} for display.

### Training Configuration

- **Optimizer**: SGD with Momentum
  - Momentum = 0.9 (helps escape local minima)
- **Learning Rate**: 0.5 (higher rate for faster convergence on this simple problem)
- **Max Epochs**: 10000 (typically converges around epoch 1000-1500)
- **Batch Size**: 4 (all samples - full batch gradient descent)
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Enabled (stops when 100% accuracy achieved after epoch 100)

### Key Components

1. **Backpropagation**: Custom implementation calculating gradients layer-by-layer
2. **SGD with Momentum**: Helps overcome local minima common in XOR training
3. **MSE Loss**: Simple and effective for binary classification
4. **Metrics Tracking**: Loss and accuracy recorded every epoch
5. **Visualization**: Training curves and decision boundary plots

## How to Run

### Prerequisites

Ensure you have the required dependencies installed:
```bash
  pip install -r requirements.txt
```

Required packages:
- numpy
- matplotlib
- pyyaml

### Running the Training Script

**From the project root directory:**
```bash
  python experiments/exercise_3/xor_mlp/train_xor.py
```

**Or from the exercise directory:**
```bash
  cd experiments/exercise_3/xor_mlp
  python train_xor.py
```

## Output Files

After successful training, the following files are generated:

### 1. Trained Weights
**Location**: `experiments/exercise_3/xor_mlp/outputs/weights/xor_mlp_weights.npz`

Contains:
- Network topology
- All layer weights (including biases)
- Activation function type
- Dropout rate

Can be loaded later for inference without retraining.

### 2. Training History Plot
**Location**: `experiments/exercise_3/xor_mlp/outputs/plots/xor_training_history.png`

Visualizes:
- **Left panel**: Loss (MSE) over epochs (should decrease smoothly)
- **Right panel**: Accuracy over epochs (should reach 100%)

Useful for diagnosing convergence issues or overfitting.

### 3. Decision Boundary Plot
**Location**: `experiments/exercise_3/xor_mlp/outputs/plots/xor_decision_boundary.png`

Shows:
- 2D visualization of the learned decision boundary
- Color regions representing different classes
- Training points overlaid with their true labels
- Demonstrates how the network creates a non-linear separation

## Configuration

Training parameters can be modified in `config.yaml`:

```yaml
network:
  architecture:
    topology: [2, 4, 1]      # Can experiment with different hidden layer sizes
    activation_type: "SIGMOID"  # Try: RELU, TANH (but SIGMOID works best)
    dropout_rate: 0.0         # No dropout needed for this simple problem

training:
  learning_rate: 0.5         # Lower (0.1-0.3) if training is unstable
  epochs: 10000              # Maximum iterations
  batch_size: 4              # Full batch for this tiny dataset
  optimizer:
    type: "sgd_momentum"     # Momentum helps escape local minima
    momentum: 0.9            # High momentum for smooth convergence
```

## Expected Results

### Convergence
- **Typical convergence**: 1000-1500 epochs
- **Final accuracy**: 100% (all 4 samples correctly classified)
- **Final loss**: ~0.15-0.25 (MSE varies based on initialization)

### Performance Indicators

✅ **Success**: 
- Accuracy reaches 100% consistently
- Loss decreases smoothly without oscillations
- Decision boundary shows clear non-linear separation
- Raw outputs cluster around 0.2-0.3 for class 0, and 0.6-0.7 for class 1

❌ **Failure signs**:
- Accuracy stuck at 50% or 75% (local minimum)
- Loss oscillating or increasing
- Training takes >3000 epochs without improvement
- Raw outputs stuck at 0.5 (network not learning)