"""
Visualization and Analysis for Training Results
Plots training curves, noise curriculum effects, and model comparisons.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
import glob
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re # Added import for regular expressions

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title='Confusion Matrix'):
    """
    Plots the confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names (e.g., ['No-Speech', 'Speech']).
        save_path: Optional path to save the figure.
        title: Title of the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion Matrix saved to {save_path}")
    else:
        plt.show()

def plot_training_curves(results_path, save_path=None):
    """
    Plot training and test accuracy curves with noise curriculum overlay.
    
    Args:
        results_path: Path to JSON results file
        save_path: Optional path to save figure
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    model_type = results['model_type']
    
    if model_type in ['CNN', 'RNN']:
        # Extract data
        epochs = [e['epoch'] for e in results['epochs']]
        train_acc = [e['train_acc'] for e in results['epochs']]
        test_acc = [e['test_acc'] for e in results['epochs']]
        train_loss = [e['train_loss'] for e in results['epochs']]
        
        # Get SNR levels for curriculum overlay
        snr_values = [e['snr_db'] for e in results['epochs']]
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Accuracy
        ax1.plot(epochs, train_acc, label='Train Accuracy', linewidth=2, color='blue')
        ax1.plot(epochs, test_acc, label='Test Accuracy', linewidth=2, color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title(f'{model_type} Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss
        ax2.plot(epochs, train_loss, linewidth=2, color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Noise Curriculum (SNR)
        snr_display = [s if s != float('inf') else 100 for s in snr_values]
        ax3.plot(epochs, snr_display, linewidth=2, color='orange', marker='o')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('SNR (dB)')
        ax3.set_title('Noise Curriculum (higher = less noise)')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 105])
        
        # Add curriculum stage markers
        unique_snr = []
        stage_epochs = []
        for i, snr in enumerate(snr_values):
            if i == 0 or snr != snr_values[i-1]:
                unique_snr.append(snr)
                stage_epochs.append(i)
        
        for epoch in stage_epochs[1:]:  # Skip first
            ax1.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5)
            ax2.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()
    
    elif model_type == 'SVM':
        # Extract data for SVM stages
        stages = [s['stage'] for s in results['curriculum_stages']]
        train_acc = [s['train_acc'] for s in results['curriculum_stages']]
        test_acc = [s['test_acc'] for s in results['curriculum_stages']]
        snr_levels = [s['snr_db'] for s in results['curriculum_stages']]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Accuracy by stage
        x = np.arange(len(stages))
        width = 0.35
        ax1.bar(x - width/2, train_acc, width, label='Train Accuracy', color='blue', alpha=0.7)
        ax1.bar(x + width/2, test_acc, width, label='Test Accuracy', color='red', alpha=0.7)
        ax1.set_xlabel('Curriculum Stage')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('SVM Performance Across Curriculum Stages')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Stage {s}' for s in stages])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: SNR levels
        snr_display = [s if s != float('inf') else 100 for s in snr_levels]
        ax2.bar(x, snr_display, color='orange', alpha=0.7)
        ax2.set_xlabel('Curriculum Stage')
        ax2.set_ylabel('SNR (dB)')
        ax2.set_title('Noise Level by Stage (higher = less noise)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Stage {s}' for s in stages])
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        else:
            plt.show()

def compare_models(comparison_path, save_path=None):
    """
    Compare performance of multiple models.
    
    Args:
        comparison_path: Path to model comparison JSON file
        save_path: Optional path to save figure
    """
    with open(comparison_path, 'r') as f:
        results = json.load(f)
    
    # Extract final test accuracies
    model_names = []
    final_accuracies = []
    
    for model_type, model_results in results.items():
        model_names.append(model_type)
        
        if model_type in ['CNN', 'RNN']:
            # Get final epoch accuracy
            final_acc = model_results['epochs'][-1]['test_acc']
        else:  # SVM
            # Get final stage accuracy
            final_acc = model_results['curriculum_stages'][-1]['test_acc']
        
        final_accuracies.append(final_acc)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax.bar(model_names, final_accuracies, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Comparison - Final Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(final_accuracies) * 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def plot_curriculum_impact(results_path, save_path=None):
    """
    Analyze and plot the impact of noise curriculum on model performance.
    
    Args:
        results_path: Path to JSON results file
        save_path: Optional path to save figure
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    model_type = results['model_type']
    
    if model_type not in ['CNN', 'RNN']:
        print("Curriculum impact plot only available for CNN and RNN models")
        return
    
    # Group epochs by SNR level
    snr_groups = {}
    for epoch_data in results['epochs']:
        snr = epoch_data['snr_db']
        if snr == float('inf') or snr is None:
            snr = 'Clean'
        else:
            snr = f'{snr} dB'

        if snr not in snr_groups:
            snr_groups[snr] = {'train_acc': [], 'test_acc': []}

        snr_groups[snr]['train_acc'].append(epoch_data['train_acc'])
        snr_groups[snr]['test_acc'].append(epoch_data['test_acc'])

    # Calculate average performance at each noise level
    snr_labels = []
    avg_train_acc = []
    avg_test_acc = []

    # Sort by SNR (Clean first, then descending)
    def snr_sort_key(x):
        if x == 'Clean':
            return float('inf')
        try:
            return float(x.split()[0])
        except (ValueError, AttributeError, IndexError):
            return -float('inf')  # Put invalid entries at the end

    sorted_snr = sorted(snr_groups.keys(), key=snr_sort_key, reverse=True)
    
    for snr in sorted_snr:
        snr_labels.append(snr)
        avg_train_acc.append(np.mean(snr_groups[snr]['train_acc']))
        avg_test_acc.append(np.mean(snr_groups[snr]['test_acc']))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(snr_labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, avg_train_acc, width, label='Train Accuracy', 
                   color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, avg_test_acc, width, label='Test Accuracy', 
                   color='red', alpha=0.7)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Noise Level', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'{model_type} Performance vs Noise Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

def generate_report(results_path):
    """
    Generate a text report summarizing training results.
    
    Args:
        results_path: Path to JSON results file
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    model_type = results['model_type']
    test_fold = results['test_fold']
    
    print("\n" + "=" * 70)
    print(f"TRAINING REPORT - {model_type} Model (Test Fold {test_fold})")
    print("=" * 70)
    
    if model_type in ['CNN', 'RNN']:
        # Get initial and final performance
        initial = results['epochs'][0]
        final = results['epochs'][-1]
        
        # Find best epoch
        best_epoch = max(results['epochs'], key=lambda x: x['test_acc'])
        
        print(f"\nTotal Epochs: {len(results['epochs'])}")
        print(f"\nInitial Performance (Epoch 0):")
        print(f"  Train Accuracy: {initial['train_acc']:.2f}%")
        print(f"  Test Accuracy: {initial['test_acc']:.2f}%")
        
        print(f"\nFinal Performance (Epoch {final['epoch']}):")
        print(f"  Train Accuracy: {final['train_acc']:.2f}%")
        print(f"  Test Accuracy: {final['test_acc']:.2f}%")
        print(f"  Final Loss: {final['train_loss']:.4f}")
        
        print(f"\nBest Performance (Epoch {best_epoch['epoch']}):")
        print(f"  Test Accuracy: {best_epoch['test_acc']:.2f}%")
        print(f"  SNR at Best: {best_epoch['snr_db']} dB")
        
        # Curriculum stages summary
        print(f"\nNoise Curriculum Summary:")
        snr_stages = {}
        for epoch in results['epochs']:
            snr = epoch['snr_db']
            if snr not in snr_stages:
                snr_stages[snr] = []
            snr_stages[snr].append(epoch['test_acc'])

        # Sort SNR values, handling None and inf
        sorted_snrs = sorted(snr_stages.keys(),
                            key=lambda x: float('inf') if x is None or x == float('inf') else x,
                            reverse=True)

        for snr in sorted_snrs:
            avg_acc = np.mean(snr_stages[snr])
            if snr == float('inf') or snr is None:
                print(f"  Clean Audio: {avg_acc:.2f}% average test accuracy")
            else:
                print(f"  SNR {snr} dB: {avg_acc:.2f}% average test accuracy")
    
    else:  # SVM
        print(f"\nTotal Curriculum Stages: {len(results['curriculum_stages'])}")
        
        for stage in results['curriculum_stages']:
            snr = stage['snr_db']
            snr_label = 'Clean' if snr == float('inf') else f'{snr} dB'
            print(f"\nStage {stage['stage']} ({snr_label}):")
            print(f"  Train Accuracy: {stage['train_acc']:.2f}%")
            print(f"  Test Accuracy: {stage['test_acc']:.2f}%")
        
        # Best stage
        best_stage = max(results['curriculum_stages'], key=lambda x: x['test_acc'])
        print(f"\nBest Performance:")
        print(f"  Stage {best_stage['stage']}")
        print(f"  Test Accuracy: {best_stage['test_acc']:.2f}%")
    
    print("\n" + "=" * 70)

def find_latest_results_dir(base_dir='results'): # Changed name and default base_dir
    """
    Find the directory containing the most recent results.

    Args:
        base_dir: Directory to search for result directories.

    Returns:
        Path to the latest results directory or None if not found.
    """
    all_run_dirs = [d for d in Path(base_dir).iterdir() if d.is_dir() and 'speechcnn' in d.name]
    
    if not all_run_dirs:
        return None

    # Sort by timestamp in directory name, most recent first
    def get_timestamp_from_dir_name(dir_path):
        match = re.search(r'(\d{8}_\d{6})', dir_path.name)
        if match:
            return match.group(1)
        return "" # Return empty string if no timestamp found, will be sorted to beginning

    all_run_dirs.sort(key=get_timestamp_from_dir_name, reverse=True)
    return all_run_dirs[0]

def main():
    """Main visualization execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--results-dir', type=str, help='Path to results directory (auto-detects latest if not provided)')
    parser.add_argument('--comparison', type=str, help='Path to comparison JSON file')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save plots (default: results/)')
    parser.add_argument('--report', action='store_true', help='Generate text report')

    args = parser.parse_args()

    # Auto-detect latest results directory if not provided
    run_dir = None
    if args.results_dir:
        run_dir = Path(args.results_dir)
    else:
        run_dir = find_latest_results_dir(base_dir=args.save_dir) # Use save_dir as base for finding
        if run_dir:
            print(f"Auto-detected latest results directory: {run_dir}")
        else:
            print("No results directories found. Please specify --results-dir")
            return

    # Check if results JSON exists in the run_dir
    results_json_path = list(run_dir.glob('*_results_*.json'))
    if not results_json_path:
        print(f"No results JSON file found in {run_dir}")
        return
    results_json_path = results_json_path[0] # Take the first one found

    # Load results JSON to get model_type etc.
    with open(results_json_path, 'r') as f:
        results = json.load(f)
        model_type = results['model_type'].lower()
        test_fold = results.get('test_fold', 'N/A') # Use N/A if not found

    # Create subfolder (if not already existing from auto-detection)
    # The run_dir already handles this, so we just use it directly.
    
    # Generate report
    if args.report:
        generate_report(results_json_path)

    # Plot training curves
    plot_training_curves(
        results_json_path,
        save_path=run_dir / f'{model_type}_training_curves.png'
    )

    # Plot curriculum impact (CNN/RNN only)
    if model_type in ['speechcnn', 'cnn', 'rnn']: # Adjust for 'speechcnn'
        plot_curriculum_impact(
            results_json_path,
            save_path=run_dir / f'{model_type}_curriculum_impact.png'
        )

    # Load and plot confusion matrix
    try:
        test_preds = np.load(run_dir / 'test_preds.npy')
        test_labels = np.load(run_dir / 'test_labels.npy')
        pure_noise_preds = np.load(run_dir / 'pure_noise_preds.npy')
        pure_noise_labels = np.load(run_dir / 'pure_noise_labels.npy')

        all_preds = np.concatenate((test_preds, pure_noise_preds))
        all_labels = np.concatenate((test_labels, pure_noise_labels))
        class_names = ['No-Speech', 'Speech']

        plot_confusion_matrix(
            all_labels, all_preds, class_names,
            save_path=run_dir / f'{model_type}_confusion_matrix.png',
            title=f'{model_type} Confusion Matrix (Combined Test Sets)'
        )
    except FileNotFoundError:
        print(f"Prediction/Label .npy files not found in {run_dir}. Skipping confusion matrix plot.")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")


    print(f"\nPlots saved to {run_dir}/")

    if args.comparison:
        # Create subfolder for comparisons
        comparison_dir = run_dir.parent / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # Use run_dir's parent
        comparison_dir.mkdir(exist_ok=True)

        # Compare models
        compare_models(
            Path(args.comparison), # Ensure it's a Path object
            save_path=comparison_dir / 'model_comparison.png'
        )

        print(f"\nComparison plots saved to {comparison_dir}/")
