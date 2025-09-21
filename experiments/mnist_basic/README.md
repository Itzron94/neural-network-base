# MNIST Basic Experiment

## Description
Basic MNIST digit classification experiment using a 2-layer neural network with ReLU activation.

## Network Architecture
- Input: 784 neurons (28x28 flattened images)
- Hidden: 128 neurons with ReLU activation
- Output: 10 neurons (digits 0-9) with Softmax

## Configuration
- Learning Rate: 0.001
- Batch Size: 64
- Epochs: 100
- Dropout: 0.1
- Early Stopping: Enabled (patience=15)

## Expected Results
- Training Accuracy: ~95-98%
- Validation Accuracy: ~94-97%
- Training Time: ~2-5 minutes

## Running the Experiment
```bash
cd experiments/mnist_basic
python train.py
```

## Output Files
- `outputs/weights/`: Trained model weights
- `outputs/logs/`: Training metrics and logs
- `outputs/plots/`: Training curves and analysis plots