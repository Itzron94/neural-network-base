"""
Metrics analysis and visualization for neural network experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import seaborn as sns


class MetricsAnalyzer:
    """
    Analyzes and visualizes training metrics from neural network experiments.
    
    Provides tools for loading metrics data, generating plots, and performing
    statistical analysis of training performance.
    """
    
    def __init__(self, save_plots: bool = True, plots_path: str = "outputs/plots/"):
        """
        Initialize metrics analyzer.
        
        Args:
            save_plots: Whether to save generated plots
            plots_path: Directory to save plots
        """
        self.save_plots = save_plots
        self.plots_path = plots_path
        
        # Configure matplotlib and seaborn
        plt.style.use('default')
        sns.set_palette("husl")
        
        if save_plots:
            os.makedirs(plots_path, exist_ok=True)
    
    def load_metrics_from_file(self, metrics_file: str) -> Dict[str, Any]:
        """
        Load metrics data from JSON file.
        
        Args:
            metrics_file: Path to metrics JSON file
            
        Returns:
            Dictionary containing metrics data
        """
        if not os.path.exists(metrics_file):
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
        
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def plot_training_curves(self, 
                           metrics_data: Dict[str, Any], 
                           title: Optional[str] = None) -> plt.Figure:
        """
        Plot training and validation curves.
        
        Args:
            metrics_data: Metrics data from load_metrics_from_file
            title: Custom title for the plot
            
        Returns:
            matplotlib Figure object
        """
        epoch_metrics = metrics_data.get('epoch_metrics', [])
        if not epoch_metrics:
            raise ValueError("No epoch metrics found in data")
        
        # Extract data
        epochs = [m['epoch'] for m in epoch_metrics]
        losses = [m['loss'] for m in epoch_metrics]
        accuracies = [m['accuracy'] for m in epoch_metrics if m['accuracy'] is not None]
        val_losses = [m['val_loss'] for m in epoch_metrics if m['val_loss'] is not None]
        val_accuracies = [m['val_accuracy'] for m in epoch_metrics if m['val_accuracy'] is not None]
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(epochs, losses, label='Training Loss', linewidth=2)
        if val_losses:
            val_epochs = [m['epoch'] for m in epoch_metrics if m['val_loss'] is not None]
            axes[0].plot(val_epochs, val_losses, label='Validation Loss', linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy if available
        if accuracies:
            acc_epochs = [m['epoch'] for m in epoch_metrics if m['accuracy'] is not None]
            axes[1].plot(acc_epochs, accuracies, label='Training Accuracy', linewidth=2)
            
            if val_accuracies:
                val_acc_epochs = [m['epoch'] for m in epoch_metrics if m['val_accuracy'] is not None]
                axes[1].plot(val_acc_epochs, val_accuracies, label='Validation Accuracy', linewidth=2)
            
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Training Accuracy Over Time')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No accuracy data available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Accuracy Not Tracked')
        
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            experiment_name = metrics_data.get('experiment_name', 'Unknown')
            fig.suptitle(f'Training Metrics - {experiment_name}', fontsize=16)
        
        plt.tight_layout()
        
        if self.save_plots:
            plot_name = f"{metrics_data.get('experiment_name', 'experiment')}_training_curves.png"
            plt.savefig(os.path.join(self.plots_path, plot_name), dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_learning_rate_schedule(self, metrics_data: Dict[str, Any]) -> Optional[plt.Figure]:
        """
        Plot learning rate schedule if available.
        
        Args:
            metrics_data: Metrics data from load_metrics_from_file
            
        Returns:
            matplotlib Figure object or None if no LR data
        """
        epoch_metrics = metrics_data.get('epoch_metrics', [])
        lr_data = [m['learning_rate'] for m in epoch_metrics if m['learning_rate'] is not None]
        
        if not lr_data:
            return None
        
        epochs = [m['epoch'] for m in epoch_metrics if m['learning_rate'] is not None]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, lr_data, linewidth=2, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        if self.save_plots:
            plot_name = f"{metrics_data.get('experiment_name', 'experiment')}_learning_rate.png"
            plt.savefig(os.path.join(self.plots_path, plot_name), dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_convergence(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze training convergence characteristics.
        
        Args:
            metrics_data: Metrics data from load_metrics_from_file
            
        Returns:
            Dictionary with convergence analysis
        """
        epoch_metrics = metrics_data.get('epoch_metrics', [])
        if len(epoch_metrics) < 5:
            return {"error": "Not enough data for convergence analysis"}
        
        losses = [m['loss'] for m in epoch_metrics]
        epochs = [m['epoch'] for m in epoch_metrics]
        
        # Calculate convergence metrics
        initial_loss = losses[0]
        final_loss = losses[-1]
        min_loss = min(losses)
        min_loss_epoch = epochs[losses.index(min_loss)]
        
        # Loss reduction rate
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        
        # Find convergence point (where loss stabilizes)
        convergence_epoch = self._find_convergence_point(losses)
        
        # Calculate smoothness (variance in loss changes)
        loss_changes = [abs(losses[i] - losses[i-1]) for i in range(1, len(losses))]
        smoothness = 1 / (1 + np.std(loss_changes))  # Higher is smoother
        
        analysis = {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "min_loss": min_loss,
            "min_loss_epoch": min_loss_epoch,
            "loss_reduction_percent": loss_reduction,
            "convergence_epoch": convergence_epoch,
            "training_smoothness": smoothness,
            "overfitting_detected": self._detect_overfitting(metrics_data),
            "convergence_quality": self._assess_convergence_quality(losses)
        }
        
        return analysis
    
    def compare_experiments(self, 
                          experiment_files: List[str], 
                          metric: str = 'loss') -> plt.Figure:
        """
        Compare multiple experiments on the same plot.
        
        Args:
            experiment_files: List of paths to metrics JSON files
            metric: Metric to compare ('loss' or 'accuracy')
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for file_path in experiment_files:
            try:
                data = self.load_metrics_from_file(file_path)
                epoch_metrics = data.get('epoch_metrics', [])
                
                if metric == 'loss':
                    values = [m['loss'] for m in epoch_metrics]
                    ylabel = 'Loss'
                elif metric == 'accuracy':
                    values = [m['accuracy'] for m in epoch_metrics if m['accuracy'] is not None]
                    ylabel = 'Accuracy'
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                epochs = [m['epoch'] for m in epoch_metrics[:len(values)]]
                experiment_name = data.get('experiment_name', Path(file_path).stem)
                
                ax.plot(epochs, values, label=experiment_name, linewidth=2)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Experiment Comparison - {metric.title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if self.save_plots:
            plot_name = f"experiment_comparison_{metric}.png"
            plt.savefig(os.path.join(self.plots_path, plot_name), dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_report(self, metrics_data: Dict[str, Any]) -> str:
        """
        Generate a text report of training metrics.
        
        Args:
            metrics_data: Metrics data from load_metrics_from_file
            
        Returns:
            Formatted text report
        """
        summary = metrics_data.get('summary', {})
        convergence = self.analyze_convergence(metrics_data)
        
        report = f"""
TRAINING REPORT
===============

Experiment: {summary.get('experiment_name', 'Unknown')}
Total Epochs: {summary.get('total_epochs', 'N/A')}
Training Time: {summary.get('training_time', 'N/A')}

FINAL METRICS
-------------
Final Loss: {summary.get('final_loss', 'N/A'):.6f}
Best Loss: {summary.get('best_loss', 'N/A'):.6f}
Final Accuracy: {summary.get('final_accuracy', 'N/A'):.4f}
Best Accuracy: {summary.get('best_accuracy', 'N/A'):.4f}

CONVERGENCE ANALYSIS
--------------------
Loss Reduction: {convergence.get('loss_reduction_percent', 'N/A'):.2f}%
Convergence Epoch: {convergence.get('convergence_epoch', 'N/A')}
Training Smoothness: {convergence.get('training_smoothness', 'N/A'):.4f}
Overfitting Detected: {convergence.get('overfitting_detected', 'N/A')}
Convergence Quality: {convergence.get('convergence_quality', 'N/A')}
"""
        
        if self.save_plots:
            report_name = f"{summary.get('experiment_name', 'experiment')}_report.txt"
            with open(os.path.join(self.plots_path, report_name), 'w') as f:
                f.write(report)
        
        return report
    
    def _find_convergence_point(self, losses: List[float], window: int = 10) -> Optional[int]:
        """Find the epoch where training converged."""
        if len(losses) < window * 2:
            return None
        
        for i in range(window, len(losses) - window):
            recent_std = np.std(losses[i:i+window])
            if recent_std < 0.001:  # Very low variance
                return i
        
        return None
    
    def _detect_overfitting(self, metrics_data: Dict[str, Any]) -> bool:
        """Detect if overfitting occurred during training."""
        epoch_metrics = metrics_data.get('epoch_metrics', [])
        
        val_losses = [m['val_loss'] for m in epoch_metrics if m['val_loss'] is not None]
        train_losses = [m['loss'] for m in epoch_metrics[:len(val_losses)]]
        
        if len(val_losses) < 10:
            return False
        
        # Check if validation loss starts increasing while training loss decreases
        recent_val = val_losses[-5:]
        recent_train = train_losses[-5:]
        
        val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
        train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
        
        return val_trend > 0 and train_trend < 0
    
    def _assess_convergence_quality(self, losses: List[float]) -> str:
        """Assess the quality of convergence."""
        if len(losses) < 20:
            return "Insufficient data"
        
        # Check final 25% of training
        final_quarter = losses[-len(losses)//4:]
        stability = np.std(final_quarter) / np.mean(final_quarter)
        
        if stability < 0.01:
            return "Excellent"
        elif stability < 0.05:
            return "Good"
        elif stability < 0.1:
            return "Fair"
        else:
            return "Poor"