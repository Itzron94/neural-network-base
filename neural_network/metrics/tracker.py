"""
Metrics tracking for neural network training.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    loss: float
    accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsTracker:
    """
    Tracks and stores metrics during neural network training.
    
    Provides functionality to record training metrics, save them to files,
    and export for later analysis.
    """
    
    def __init__(self, experiment_name: str, save_path: str = "outputs/logs/"):
        """
        Initialize metrics tracker.
        
        Args:
            experiment_name: Name of the experiment
            save_path: Directory to save metrics files
        """
        self.experiment_name = experiment_name
        self.save_path = save_path
        self.start_time = datetime.now()
        
        # Metrics storage
        self.epoch_metrics: List[EpochMetrics] = []
        self.batch_losses: List[float] = []
        self.gradient_norms: List[float] = []
        
        # Training state
        self.current_epoch = 0
        self.total_epochs = 0
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        
        # Create save directory
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(save_path, f"{experiment_name}_metrics.json")
        self._save_experiment_info()
    
    def start_epoch(self, epoch: int, total_epochs: int) -> None:
        """Start tracking a new epoch."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        self.batch_losses.clear()
    
    def record_batch_loss(self, loss: float) -> None:
        """Record loss for a single batch."""
        self.batch_losses.append(loss)
    
    def record_gradient_norm(self, grad_norm: float) -> None:
        """Record gradient norm for monitoring gradient flow."""
        self.gradient_norms.append(grad_norm)
    
    def end_epoch(self, 
                  loss: float, 
                  accuracy: Optional[float] = None,
                  val_loss: Optional[float] = None,
                  val_accuracy: Optional[float] = None,
                  learning_rate: Optional[float] = None) -> None:
        """
        End current epoch and record metrics.
        
        Args:
            loss: Training loss for the epoch
            accuracy: Training accuracy for the epoch
            val_loss: Validation loss for the epoch
            val_accuracy: Validation accuracy for the epoch
            learning_rate: Current learning rate
        """
        epoch_metrics = EpochMetrics(
            epoch=self.current_epoch,
            loss=loss,
            accuracy=accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            learning_rate=learning_rate
        )
        
        self.epoch_metrics.append(epoch_metrics)
        
        # Update best metrics
        if accuracy is not None and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        # Save metrics to file
        self._save_metrics()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of training metrics."""
        if not self.epoch_metrics:
            return {"status": "No training data recorded"}
        
        losses = [m.loss for m in self.epoch_metrics]
        accuracies = [m.accuracy for m in self.epoch_metrics if m.accuracy is not None]
        
        summary = {
            "experiment_name": self.experiment_name,
            "total_epochs": len(self.epoch_metrics),
            "training_time": str(datetime.now() - self.start_time),
            "final_loss": losses[-1] if losses else None,
            "best_loss": min(losses) if losses else None,
            "final_accuracy": accuracies[-1] if accuracies else None,
            "best_accuracy": max(accuracies) if accuracies else None,
            "loss_improvement": losses[0] - losses[-1] if len(losses) > 1 else 0,
            "convergence_epoch": self._find_convergence_epoch(),
            "avg_gradient_norm": np.mean(self.gradient_norms) if self.gradient_norms else None
        }
        
        return summary
    
    def get_epoch_data(self) -> List[Dict[str, Any]]:
        """Get all epoch data as list of dictionaries."""
        return [
            {
                "epoch": m.epoch,
                "loss": m.loss,
                "accuracy": m.accuracy,
                "val_loss": m.val_loss,
                "val_accuracy": m.val_accuracy,
                "learning_rate": m.learning_rate,
                "timestamp": m.timestamp
            }
            for m in self.epoch_metrics
        ]
    
    def export_to_csv(self, filename: Optional[str] = None) -> str:
        """
        Export metrics to CSV file.
        
        Args:
            filename: Custom filename, if not provided uses default
            
        Returns:
            Path to the created CSV file
        """
        if filename is None:
            filename = f"{self.experiment_name}_metrics.csv"
        
        csv_path = os.path.join(self.save_path, filename)
        
        if not self.epoch_metrics:
            raise ValueError("No metrics data to export")
        
        # Create CSV content
        headers = ["epoch", "loss", "accuracy", "val_loss", "val_accuracy", "learning_rate", "timestamp"]
        lines = [",".join(headers)]
        
        for metrics in self.epoch_metrics:
            row = [
                str(metrics.epoch),
                str(metrics.loss),
                str(metrics.accuracy) if metrics.accuracy is not None else "",
                str(metrics.val_loss) if metrics.val_loss is not None else "",
                str(metrics.val_accuracy) if metrics.val_accuracy is not None else "",
                str(metrics.learning_rate) if metrics.learning_rate is not None else "",
                metrics.timestamp
            ]
            lines.append(",".join(row))
        
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        
        return csv_path
    
    def should_stop_early(self, patience: int = 10, min_delta: float = 0.001) -> bool:
        """
        Check if training should stop early based on validation metrics.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            
        Returns:
            True if training should stop early
        """
        if len(self.epoch_metrics) < patience + 1:
            return False
        
        # Check validation loss if available, otherwise use training loss
        recent_metrics = self.epoch_metrics[-patience-1:]
        
        if recent_metrics[0].val_loss is not None:
            best_loss = min(m.val_loss for m in recent_metrics[:-1])
            current_loss = recent_metrics[-1].val_loss
        else:
            best_loss = min(m.loss for m in recent_metrics[:-1])
            current_loss = recent_metrics[-1].loss
        
        return current_loss >= best_loss - min_delta
    
    def _find_convergence_epoch(self) -> Optional[int]:
        """Find the epoch where the model likely converged."""
        if len(self.epoch_metrics) < 10:
            return None
        
        losses = [m.loss for m in self.epoch_metrics]
        
        # Look for when loss change becomes small
        for i in range(10, len(losses)):
            recent_change = abs(losses[i] - losses[i-5]) / losses[i-5]
            if recent_change < 0.01:  # Less than 1% change
                return i
        
        return None
    
    def _save_experiment_info(self) -> None:
        """Save initial experiment information."""
        info = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "metrics": []
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
    
    def _save_metrics(self) -> None:
        """Save current metrics to JSON file."""
        data = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "last_updated": datetime.now().isoformat(),
            "summary": self.get_training_summary(),
            "epoch_metrics": self.get_epoch_data()
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)