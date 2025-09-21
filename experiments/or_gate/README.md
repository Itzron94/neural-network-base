# OR Gate Experiment

## Description
Simple logical OR gate learning experiment to demonstrate basic neural network functionality.

## Network Architecture
- Input: 2 neurons (binary inputs)
- Hidden: 4 neurons with Sigmoid activation
- Output: 1 neuron (binary output) with Sigmoid

## Configuration
- Learning Rate: 0.01
- Batch Size: 4 (all training samples)
- Epochs: 500
- Dropout: 0.0 (disabled for simple problem)
- Early Stopping: Disabled

## Training Data
| Input A | Input B | Output (A OR B) |
|---------|---------|----------------|
|    0    |    0    |       0        |
|    0    |    1    |       1        |
|    1    |    0    |       1        |
|    1    |    1    |       1        |

## Expected Results
- Training Accuracy: 100%
- Loss: < 0.01
- Convergence: ~100-200 epochs

## Running the Experiment
```bash
cd experiments/or_gate
python train.py
```

## Output Files
- `outputs/weights/`: Trained model weights
- `outputs/logs/`: Training metrics and logs
- `outputs/plots/`: Training curves and analysis plots