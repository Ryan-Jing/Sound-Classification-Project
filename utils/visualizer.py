import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import warnings

def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """Calculate the bootstrap confidence interval for a 1D array of data."""
    if len(data) == 0:
        return 0, (0, 0)
        
    means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        # Resample with replacement
        sample_indices = np.random.choice(range(len(data)), size=len(data), replace=True)
        sample = data[sample_indices]
        means[i] = np.mean(sample)
    
    mean_of_means = np.mean(means)
    
    # Calculate confidence interval
    lower_bound = np.percentile(means, (100 - ci) / 2)
    upper_bound = np.percentile(means, 100 - (100 - ci) / 2)
    
    return mean_of_means, (lower_bound, upper_bound)


def plot_aggregated_history_with_ci(histories, save_path):
    """
    Plots aggregated training and validation error with confidence intervals.
    
    Args:
        histories (list of dict): A list where each element is a history dictionary from a trial.
        save_path (str or Path): Path to save the plot image.
    """
    if not histories:
        print("No history to plot.")
        return

    # Aggregate data from all runs
    train_acc_all_runs = np.array([h['train_acc'] for h in histories])
    val_acc_all_runs = np.array([h['val_acc'] for h in histories])
    
    # Convert accuracy to error
    train_error_all_runs = 1 - train_acc_all_runs
    val_error_all_runs = 1 - val_acc_all_runs

    epochs = train_error_all_runs.shape[1]
    epochs_range = range(1, epochs + 1)
    
    # Calculate mean and CI for each epoch
    mean_train_errors, ci_train_lows, ci_train_highs = [], [], []
    mean_val_errors, ci_val_lows, ci_val_highs = [], [], []

    for i in range(epochs):
        mean_train, (low, high) = bootstrap_ci(train_error_all_runs[:, i])
        mean_train_errors.append(mean_train)
        ci_train_lows.append(low)
        ci_train_highs.append(high)

        mean_val, (low, high) = bootstrap_ci(val_error_all_runs[:, i])
        mean_val_errors.append(mean_val)
        ci_val_lows.append(low)
        ci_val_highs.append(high)

    print(f"Final Mean Training Error: {mean_train_errors[-1]:.4f} (95% CI: [{ci_train_lows[-1]:.4f}, {ci_train_highs[-1]:.4f}])")
    print(f"Final Mean Validation Error: {mean_val_errors[-1]:.4f} (95% CI: [{ci_val_lows[-1]:.4f}, {ci_val_highs[-1]:.4f}])")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Training Error Plot
    # Plot individual runs faintly
    for run_data in train_error_all_runs:
        ax1.plot(epochs_range, run_data, color='gray', alpha=0.3, linewidth=0.8)
    # Plot mean and CI
    ax1.plot(epochs_range, mean_train_errors, color='blue', linewidth=2, label='Mean Training Error')
    ax1.fill_between(epochs_range, ci_train_lows, ci_train_highs, color='blue', alpha=0.2, label='95% Bootstrap CI')
    ax1.set_title('Model Training Error Across 5 Trials')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Error')
    ax1.legend()

    # Validation Error Plot
    # Plot individual runs faintly
    for run_data in val_error_all_runs:
        ax2.plot(epochs_range, run_data, color='gray', alpha=0.3, linewidth=0.8)
    # Plot mean and CI
    ax2.plot(epochs_range, mean_val_errors, color='red', linewidth=2, label='Mean Validation Error')
    ax2.fill_between(epochs_range, ci_val_lows, ci_val_highs, color='red', alpha=0.2, label='95% Bootstrap CI')
    ax2.set_title('Model Validation Error Across 5 Trials')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Error')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Aggregated training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Computes, plots, and saves a confusion matrix.
    Handles potential UserWarning for single-label cases.
    """
    with warnings.catch_warnings():
        # Suppress the warning about single labels, as we handle it with the 'labels' param
        warnings.filterwarnings("ignore", message="A single label was found in 'y_true' and 'y_pred'.", category=UserWarning)
        
        # Ensure all potential classes are represented in the matrix
        unique_labels = np.unique(np.concatenate((y_true, y_pred)))
        all_class_indices = np.arange(len(class_names))
        
        # Use all known class indices for the matrix calculation
        cm = confusion_matrix(y_true, y_pred, labels=all_class_indices)
    
    # Normalize the confusion matrix
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_normalized = np.where(cm_sum > 0, cm.astype('float') / cm_sum, 0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt=".2f", 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False
    )
    plt.title('Normalized Confusion Matrix (Aggregated over 5 Trials)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Aggregated confusion matrix saved to {save_path}")