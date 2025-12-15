import os
import datetime
import pandas as pd
from pathlib import Path
import numpy as np

# Custom module imports
from data_loader import get_user_choices, prepare_dataloaders
from models.cnn import CNNModel
from models.rnn import RNNModel
from trainers.pytorch_trainer import train_pytorch_model
from trainers.svm_trainer import train_svm_model
from utils.visualizer import plot_aggregated_history_with_ci, plot_confusion_matrix
from utils.spectrogram_visualizer import save_example_spectrograms

def main():
    """
    Main function to run the interactive training pipeline with multiple trials.
    """
    NUM_TRIALS = 5

    # --- 1. Get User Input ---
    base_data_dir = Path('data/organized_audio_datasets')
    features, model_type = get_user_choices(base_data_dir)

    print(f"\nStarting training process for features: {', '.join(features)}")
    print(f"Using model type: {model_type}")
    print(f"Running {NUM_TRIALS} trials...")

    # --- 2. Setup Output Directory ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_str = features[0] if len(features) == 1 else 'all'
    model_name = f"{model_type}_{feature_str}_{timestamp}"
    output_dir = Path('results') / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # --- 3. Load Data ---
    print("\nLoading and preparing data...")
    dataloaders, class_to_idx = prepare_dataloaders(
        base_data_dir, features, model_type
    )
    class_names = list(class_to_idx.keys())
    num_classes = len(class_names)

    # Save the class mapping
    class_map_df = pd.DataFrame(class_to_idx.items(), columns=['class_name', 'class_id'])
    class_map_df.to_csv(output_dir / 'class_mapping.csv', index=False)
    print(f"Class mapping saved to {output_dir / 'class_mapping.csv'}")

    # --- 4. Run Trials ---
    all_histories = []
    all_test_labels = []
    all_test_preds = []

    for i in range(NUM_TRIALS):
        print(f"\n{'='*20} TRIAL {i+1}/{NUM_TRIALS} {'='*20}")

        # On the first trial, save some example spectrograms
        if i == 0 and model_type != 'SVM' and dataloaders.get('train'):
            save_example_spectrograms(dataloaders['train'], class_names, output_dir)

        if model_type in ['CNN', 'RNN']:
            # Re-initialize the model for each trial
            if model_type == 'CNN':
                model = CNNModel(num_classes=num_classes)
            else: # RNN
                model = RNNModel(input_size=128, num_classes=num_classes)

            print(f"Training {model_type} model...")
            learning_rates = {'clean': 1e-3, '20dB': 5e-4, '15dB': 2e-4, '10dB': 1e-4, '5dB': 5e-5}

            trial_model_name = f"{model_name}_trial_{i+1}"

            history, test_labels, test_preds = train_pytorch_model(
                model=model,
                dataloaders=dataloaders,
                learning_rates=learning_rates,
                output_dir=output_dir,
                num_epochs_per_level=5,
                model_name=trial_model_name,
                class_names=class_names,
                trial_num=i
            )
            all_histories.append(history)

        elif model_type == 'SVM':
            print(f"Training {model_type} model...")
            # Note: SVM training is deterministic, so multiple trials will yield the same result,
            # but we run it in a loop for consistent structure.
            test_labels, test_preds = train_svm_model(
                dataloaders=dataloaders,
                class_names=class_names
            )

        all_test_labels.append(test_labels)
        all_test_preds.append(test_preds)

    # --- 5. Aggregate, Visualize, and Save Results ---
    print(f"\n{'='*20} AGGREGATING RESULTS {'='*20}")

    # For PyTorch models, plot aggregated training history
    if model_type in ['CNN', 'RNN'] and all_histories:
        plot_aggregated_history_with_ci(all_histories, output_dir / f"{model_name}_aggregated_learning_curves.png")

    # Aggregate all labels and predictions for a final confusion matrix
    final_labels = np.concatenate(all_test_labels)
    final_preds = np.concatenate(all_test_preds)

    plot_confusion_matrix(final_labels, final_preds, class_names, output_dir / f"{model_name}_aggregated_confusion_matrix.png")

    print(f"\n✅ All trials complete. Aggregated results saved in {output_dir}")

if __name__ == '__main__':
    main()
