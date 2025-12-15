import os
import datetime
import pandas as pd
from pathlib import Path

# Custom module imports
from data_loader import get_user_choices, prepare_dataloaders
from models.cnn import CNNModel
from models.rnn import RNNModel
from trainers.pytorch_trainer import train_pytorch_model
from trainers.svm_trainer import train_svm_model
from utils.visualizer import plot_training_history, plot_confusion_matrix

def main():
    """
    Main function to run the interactive training pipeline.
    """
    # --- 1. Get User Input ---
    base_data_dir = Path('data/organized_audio_datasets')
    features, model_type = get_user_choices(base_data_dir)

    print(f"\nStarting training process for features: {', '.join(features)}")
    print(f"Using model type: {model_type}")

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

    # --- 4. Initialize and Train Model ---
    if model_type in ['CNN', 'RNN']:
        # Define learning rates for curriculum
        learning_rates = {'clean': 1e-3, '20dB': 5e-4, '15dB': 2e-4, '10dB': 1e-4, '5dB': 5e-5}

        # Initialize model
        if model_type == 'CNN':
            model = CNNModel(num_classes=num_classes)
        else: # RNN
            model = RNNModel(input_size=128, num_classes=num_classes)

        print(f"\nTraining {model_type} model with curriculum learning...")

        history, test_labels, test_preds = train_pytorch_model(
            model=model,
            dataloaders=dataloaders,
            learning_rates=learning_rates,
            output_dir=output_dir,
            num_epochs_per_level=5, # Train for 5 epochs on each noise level
            model_name=model_name,
            class_names=class_names
        )

        # --- 5. Visualize and Save Results ---
        print("\nVisualizing and saving results...")

        # Plot training history
        plot_training_history(history, output_dir / f"{model_type}_training_curves.png")

        # Plot confusion matrix
        plot_confusion_matrix(test_labels, test_preds, class_names, output_dir / f"{model_type}_confusion_matrix.png")

    elif model_type == 'SVM':
        print(f"\nTraining {model_type} model...")

        test_labels, test_preds = train_svm_model(
            dataloaders=dataloaders,
            class_names=class_names
        )

        # --- 5. Visualize and Save Results for SVM ---
        print("\nVisualizing and saving results...")
        plot_confusion_matrix(test_labels, test_preds, class_names, output_dir / f"{model_type}_confusion_matrix.png")

    print(f"\n✅ Training and evaluation complete. Results saved in {output_dir}")

if __name__ == '__main__':
    main()