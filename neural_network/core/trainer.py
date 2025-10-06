import numpy as np
from typing import List, Optional, Union, Any, Dict
from .network import NeuralNetwork
from .optimizers import OptimizerFunction, OptimizerFunctionFactory
from ..config import OptimizerConfig
from .losses import softmax_cross_entropy_with_logits
from ..utils.data_utils import calculate_score, k_fold_split, stratified_k_fold_split

class Trainer:
    def __init__(self, learning_rate, epochs, network: NeuralNetwork, loss_func,
                optimizer_config: Optional[OptimizerConfig] = None,):
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.network = network
        self.loss_f = loss_func
        if optimizer_config is None:
            optimizer_config = OptimizerConfig()
        self.optimizer: OptimizerFunction = self._create_optimizer(optimizer_config)

    def train(self, training_inputs: np.ndarray, training_labels: np.ndarray, batch_size: int = 32, verbose=False) -> None:
        if training_inputs.shape[0] != training_labels.shape[0]:
            raise ValueError("El número de muestras en 'training_inputs' y 'training_labels' debe ser el mismo.")

        num_samples = training_inputs.shape[0]
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            shuffled_inputs = training_inputs[indices]
            shuffled_labels = training_labels[indices]

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_inputs = shuffled_inputs[start_idx:end_idx]
                batch_labels = shuffled_labels[start_idx:end_idx]

                y = self.network.forward(batch_inputs, training=True)

                if np.isnan(y).any() or np.isinf(y).any():
                    print("Advertencia: y contiene valores NaN o Inf en la época", epoch)
                    continue

                batch_loss = self.loss_f(y, batch_labels)
                total_loss += batch_loss

                deltas: List[np.ndarray] = []
                deltas.insert(0, batch_loss)

                for l in range(len(self.network.layers) - 2, -1, -1):
                    current_layer = self.network.layers[l]
                    next_layer = self.network.layers[l + 1]
                    weights_next_layer = next_layer.weights[:-1, :] #Excluir el bias
                    #weights_next_layer = np.array([p.weights[:-1] for p in next_layer.perceptrons]) #Excluir el bias
                    delta_next = deltas[0]
                    delta = np.dot(delta_next, weights_next_layer.transpose()) * current_layer.get_activation_derivative()
                    deltas.insert(0, delta)

                all_gradients = []
                for l, layer in enumerate(self.network.layers):
                    inputs_to_use = batch_inputs if l == 0 else self.network.layers[l - 1].outputs
                    delta = deltas[l]
                    if delta.ndim > 0:
                        den = len(delta)
                    else:
                        den = 1
                    # Gradient for weights (excluding bias): shape (input_dim, num_perceptrons)
                    grad_weights = np.dot(inputs_to_use.T, delta) / den
                    # Gradient for bias: shape (num_perceptrons,)
                    if delta.ndim>1:
                        grad_bias = np.mean(delta, axis=0)
                    else:
                        grad_bias = np.mean(delta)
                    # Combine gradients: shape (input_dim + 1, num_perceptrons)
                    gradients = np.vstack([grad_weights, grad_bias])
                    # Split gradients for each perceptron in the layer
                    layer_gradients = [gradients[:, i] for i in range(gradients.shape[1])]
                    all_gradients.append(layer_gradients)
                self.optimizer.update_network(self.network, all_gradients, self.learning_rate)
                    
            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Época {epoch}/{self.epochs} - Pérdida: {total_loss}")

    def _create_optimizer(self, optimizer_config: OptimizerConfig) -> OptimizerFunction:
        """Create optimizer instance 
        ased on configuration."""
        # Pass all optimizer config parameters, factory will filter what each optimizer needs
        return OptimizerFunctionFactory.create(
            optimizer_config.type,
            beta1=optimizer_config.beta1,
            beta2=optimizer_config.beta2,
            epsilon=optimizer_config.epsilon,
            momentum=optimizer_config.momentum
        )

def k_fold_cross_validate(
    X: np.ndarray, 
    y: np.ndarray, 
    model: Any,
    opt_cfg: OptimizerConfig,
    loss_func,
    lr: float = 1e-3,
    batch_size: int = 4,
    epochs: int = 1000,
    k: int = 5,
    scoring: str = 'accuracy',
    shuffle: bool = True,
    random_state: int = None,
    return_models: bool = False,
    stratified: bool = False
) -> Dict[str, Union[float, List]]:
    """
    Perform k-fold cross-validation on a given dataset.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    model : object
        Model object with fit() and predict() methods
    k : int
        Number of folds
    scoring : str
        Scoring metric
    shuffle : bool
        Whether to shuffle the data
    random_state : int
        Random seed for reproducibility
    return_models : bool
        Whether to return trained models for each fold
    stratified : bool
        Whether to use stratified k-fold (for classification)
    
    Returns:
    --------
    results : Dict
        Dictionary containing cross-validation results
    """
    
    # Input validation
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    
    if k <= 1 or k > len(X):
        raise ValueError("k must be between 2 and number of samples")
    
    n_samples = len(X)
    
    # Generate folds
    if stratified:
        folds = stratified_k_fold_split(y, k, shuffle, random_state)
    else:
        folds = k_fold_split(n_samples, k, shuffle, random_state)
    
    # Initialize lists to store results
    scores = []
    models = [] if return_models else None
    fold_details = []

    #Save original weights
    og_w = model.save_weights('tmp_og_weights')
    
    for fold, (train_idx, test_idx) in enumerate(folds):
        fold_model = model
        fold_model.load_weights('tmp_og_weights.npz')
        tr = Trainer(learning_rate=lr, epochs=epochs, optimizer_config=opt_cfg,
                    loss_func=loss_func, network=fold_model)
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        tr.train(X_train, y_train, batch_size)
        
        # Make predictions
        y_pred = fold_model.forward(X_test)
        
        # Calculate score
        score = calculate_score(y_test, y_pred, scoring)
        scores.append(score)
        
        # Store model if requested
        if return_models:
            models.append(fold_model)
        
        # Store fold details
        fold_details.append({
            'fold': fold + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'score': score,
            'train_indices': train_idx,
            'test_indices': test_idx
        })
    
    # Calculate statistics
    results = {
        'fold_scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'fold_details': fold_details,
        'n_folds': k,
        'scoring': scoring
    }
    
    if return_models:
        results['models'] = models
    
    return results
