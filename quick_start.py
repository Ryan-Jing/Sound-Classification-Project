"""
Quick Start Script for Audio Classification Project
This script guides you through setup and training.
"""

import os
import sys
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def check_dataset_paths():
    """Check if dataset paths are configured."""
    print_header("Dataset Path Configuration")
    
    print("Please provide the paths to your downloaded Kaggle datasets.")
    print("Press Enter to skip a dataset if you don't have it yet.\n")
    
    datasets = {}
    
    # UrbanSound8K
    path = input("UrbanSound8K path (required): ").strip()
    if not path or not Path(path).exists():
        print("ERROR: UrbanSound8K dataset is required and path must exist!")
        sys.exit(1)
    datasets['urbansound8k'] = path
    
    # Speech dataset
    path = input("Speech Activity Detection dataset path (optional): ").strip()
    if path and Path(path).exists():
        datasets['speech'] = path
    else:
        datasets['speech'] = None
    
    # Noise datasets
    noise_paths = []
    path = input("Audio Noise Dataset path (optional): ").strip()
    if path and Path(path).exists():
        noise_paths.append(path)
    
    path = input("Noise Dataset 2 path (optional): ").strip()
    if path and Path(path).exists():
        noise_paths.append(path)
    
    datasets['noise'] = noise_paths if noise_paths else None
    
    return datasets

def organize_datasets(dataset_paths):
    """Run dataset organization."""
    print_header("Dataset Organization")
    
    from audio_dataset_organizer import AudioDatasetOrganizer
    
    print("Organizing datasets...")
    print("This may take 10-30 minutes depending on dataset size.\n")
    
    organizer = AudioDatasetOrganizer(
        target_sr=22050,
        target_duration=4.0,
        output_base_dir='organized_dataset'
    )
    
    # Process UrbanSound8K (required)
    print("Processing UrbanSound8K...")
    organizer.organize_urbansound8k(dataset_paths['urbansound8k'])
    
    # Process speech dataset (optional)
    if dataset_paths['speech']:
        print("\nProcessing Speech dataset...")
        organizer.organize_speech_dataset(dataset_paths['speech'], fold_assignment='balanced')
    else:
        print("\nSkipping Speech dataset (not provided)")
    
    # Process noise datasets (optional)
    if dataset_paths['noise']:
        print("\nProcessing Noise datasets...")
        organizer.organize_noise_datasets(dataset_paths['noise'])
    else:
        print("\nSkipping Noise datasets (not provided)")
    
    # Save metadata
    organizer.save_metadata()
    
    print("\n✓ Dataset organization complete!")
    return True

def configure_training():
    """Configure training parameters."""
    print_header("Training Configuration")
    
    print("Select model to train:")
    print("1. CNN (Convolutional Neural Network)")
    print("2. RNN/LSTM (Recurrent Neural Network)")
    print("3. SVM (Support Vector Machine)")
    print("4. All models (comparison)")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip()
    if not choice:
        choice = "1"
    
    model_map = {
        "1": "CNN",
        "2": "RNN",
        "3": "SVM",
        "4": "ALL"
    }
    model_type = model_map.get(choice, "CNN")
    
    # Test fold
    test_fold = input("\nTest fold (1-10) [default: 1]: ").strip()
    test_fold = int(test_fold) if test_fold else 1
    
    # Number of epochs (for CNN/RNN)
    if model_type in ["CNN", "RNN", "ALL"]:
        num_epochs = input("\nNumber of epochs [default: 30]: ").strip()
        num_epochs = int(num_epochs) if num_epochs else 30
    else:
        num_epochs = 6  # SVM trains at 6 curriculum stages
    
    # Batch size
    batch_size = input("\nBatch size [default: 32]: ").strip()
    batch_size = int(batch_size) if batch_size else 32
    
    # Select device
    device_choice = input("\nSelect device (auto, mps, cuda, cpu) [default: auto]: ").strip().lower()
    if not device_choice:
        device_choice = 'auto'
    
    config = {
        'model_type': model_type,
        'test_fold': test_fold,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'device': device_choice
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    confirm = input("\nProceed with training? (y/n) [default: y]: ").strip().lower()
    if confirm == 'n':
        print("Training cancelled.")
        return None
    
    return config

def run_training(config):
    """Run the training pipeline."""
    print_header(f"Training {config['model_type']} Model")
    
    from training_pipeline import train_with_curriculum, compare_all_models
    
    device = config.get('device', 'auto')
    
    if config['model_type'] == 'ALL':
        print("Training all models for comparison...\n")
        results = compare_all_models(
            organized_dataset_dir='organized_dataset',
            test_fold=config['test_fold'],
            device=device
        )
    else:
        print(f"Training {config['model_type']} model...\n")
        results = train_with_curriculum(
            model_type=config['model_type'],
            organized_dataset_dir='organized_dataset',
            test_fold=config['test_fold'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            device=device
        )
    
    print("\n✓ Training complete!")
    return results

def main():
    """Main execution."""
    print_header("Audio Classification - Quick Start")
    
    print("This script will guide you through:")
    print("1. Dataset organization and preprocessing")
    print("2. Model training with noise curriculum")
    print("\nMake sure you have downloaded the Kaggle datasets first!\n")
    
    # Check if datasets are already organized
    organized_path = Path('organized_dataset')
    if organized_path.exists():
        print(f"Found existing organized dataset at: {organized_path}")
        skip_org = input("Skip dataset organization? (y/n) [default: y]: ").strip().lower()
        if skip_org != 'n':
            print("Skipping dataset organization.")
        else:
            dataset_paths = check_dataset_paths()
            organize_datasets(dataset_paths)
    else:
        # Need to organize datasets
        dataset_paths = check_dataset_paths()
        organize_datasets(dataset_paths)
    
    # Configure and run training
    config = configure_training()
    if config:
        run_training(config)
    
    print_header("Complete!")
    print("Check the generated files:")
    print("  - Model weights: best_*.pth or svm_*.pkl")
    print("  - Training results: *_results_*.json")
    print("  - Comparison (if all models): model_comparison_*.json")
    print("\nThank you for using this audio classification system!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
