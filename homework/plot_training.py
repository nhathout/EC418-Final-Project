#!/usr/bin/env python3
"""
plot_training.py

This script loads training metrics (loss history and weight evolution)
from the specified log directory and generates corresponding plots.

Usage:
    python3 plot_training.py --log_dir ../logs/train

If --log_dir is not specified, it defaults to '../logs/train'.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_metrics(log_dir):
    """
    Load training metrics from the log directory.

    Args:
        log_dir (str): Path to the log directory.

    Returns:
        dict: A dictionary containing loss_history, weight1_history, and weight2_history.
    """
    metrics = {}
    try:
        loss_path = os.path.join(log_dir, 'loss_history.npy')
        weight1_path = os.path.join(log_dir, 'weight1_history.npy')
        weight2_path = os.path.join(log_dir, 'weight2_history.npy')

        metrics['loss_history'] = np.load(loss_path)
        metrics['weight1_history'] = np.load(weight1_path)
        metrics['weight2_history'] = np.load(weight2_path)

        print(f"Successfully loaded metrics from {log_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ensure that the training script has been run and the metrics are saved.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading metrics: {e}")
        exit(1)
    
    return metrics

def plot_loss(loss_history, log_dir):
    """
    Plot the training loss over epochs.

    Args:
        loss_history (np.ndarray): Array of loss values per epoch.
        log_dir (str): Directory to save the loss plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(log_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss curve saved to {loss_plot_path}")

def plot_weights(weight1_history, weight2_history, log_dir):
    """
    Plot the evolution of weight1 and weight2 over epochs.

    Args:
        weight1_history (np.ndarray): Array of weight1 values per epoch.
        weight2_history (np.ndarray): Array of weight2 values per epoch.
        log_dir (str): Directory to save the weights plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(weight1_history) + 1), weight1_history, label='Weight1', color='red')
    plt.plot(range(1, len(weight2_history) + 1), weight2_history, label='Weight2', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.title('Weights Evolution Over Epochs')
    plt.legend()
    plt.grid(True)
    weights_plot_path = os.path.join(log_dir, 'weights_evolution.png')
    plt.savefig(weights_plot_path)
    plt.close()
    print(f"Weights evolution plot saved to {weights_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument('--log_dir', type=str, default='../logs/train',
                        help='Directory where training metrics are saved (default: ../logs/train)')
    args = parser.parse_args()

    log_dir = args.log_dir

    if not os.path.isdir(log_dir):
        print(f"Error: The specified log directory '{log_dir}' does not exist.")
        exit(1)

    # Load metrics
    metrics = load_metrics(log_dir)

    # Plot loss history
    plot_loss(metrics['loss_history'], log_dir)

    # Plot weights evolution
    plot_weights(metrics['weight1_history'], metrics['weight2_history'], log_dir)

    print("All plots have been generated successfully.")

if __name__ == '__main__':
    main()
