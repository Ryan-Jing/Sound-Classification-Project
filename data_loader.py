import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import random
import librosa
import warnings

def get_user_choices(base_data_dir):
    """
    Interactively prompts the user to select features and model type.
    """
    # --- Feature Selection ---
    print("Please select the features to train on:")
    available_features = sorted([d.name for d in base_data_dir.iterdir() if d.is_dir()])
    for i, feature in enumerate(available_features):
        print(f"  {i+1}: {feature}")
    print(f"  {len(available_features)+1}: all")

    while True:
        try:
            choice = input(f"Enter number(s) (e.g., '1' or '1,3,4' or '{len(available_features)+1}' for all): ")
            if choice == str(len(available_features) + 1):
                selected_features = ['all']
                break

            indices = [int(i.strip()) - 1 for i in choice.split(',')]
            if all(0 <= i < len(available_features) for i in indices):
                selected_features = [available_features[i] for i in indices]
                break
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

    # --- Model Selection ---
    print("\nPlease select the model type:")
    model_types = ['CNN', 'RNN', 'SVM']
    for i, model_name in enumerate(model_types):
        print(f"  {i+1}: {model_name}")

    while True:
        try:
            choice = int(input(f"Enter number (1-{len(model_types)}): "))
            if 1 <= choice <= len(model_types):
                selected_model = model_types[choice - 1]
                break
            else:
                print("Invalid number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return selected_features, selected_model

class AudioDataset(Dataset):
    """
    PyTorch Dataset for loading audio files and converting them to spectrograms.
    Uses librosa for more stable audio loading.
    """
    def __init__(self, file_paths, labels, target_sr, n_mels, n_fft, hop_length):
        self.file_paths = file_paths
        self.labels = labels
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

            waveform = torch.from_numpy(waveform).unsqueeze(0)

            mel_spec = self.mel_spectrogram_transform(waveform)
            mel_spec_db = self.amplitude_to_db(mel_spec)

            return mel_spec_db, label
        except Exception as e:
            # print(f"Error loading {audio_path}: {e}")
            # Return a dummy tensor and label if a file is corrupt
            return torch.zeros((1, self.n_mels, 100)), -1


def prepare_dataloaders(base_dir, features, model_type, batch_size=32, test_size=0.1, val_size=0.1):
    target_sr = 16000
    n_mels = 128
    n_fft = 2048
    hop_length = 512

    all_available_classes = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])

    selected_classes = features
    if 'all' in features:
        selected_classes = all_available_classes

    # --- Negative Sampling Logic ---
    # If a subset of classes is selected, create a negative "unknown" class
    unknown_class_files = []
    if 'all' not in features and len(selected_classes) > 0:
        unselected_classes = [c for c in all_available_classes if c not in selected_classes]

        # Determine number of samples for the unknown class (average of selected classes)
        num_samples_per_selected_class = []
        for class_name in selected_classes:
            class_dir = base_dir / class_name
            audio_dirs = [class_dir / 'clean'] + [class_dir / f'{snr}dB' for snr in [5, 10, 15, 20]]
            num_samples_per_selected_class.append(sum(len(list(d.glob('*.wav'))) for d in audio_dirs if d.exists()))

        avg_samples = int(np.mean(num_samples_per_selected_class)) if num_samples_per_selected_class else 0

        # Gather all files from unselected classes
        potential_unknown_files = []
        for class_name in unselected_classes:
            class_dir = base_dir / class_name
            audio_dirs = [class_dir / 'clean'] + [class_dir / f'{snr}dB' for snr in [5, 10, 15, 20]]
            for d in audio_dirs:
                if d.exists():
                    potential_unknown_files.extend(list(d.glob('*.wav')))

        # Randomly sample to create a balanced "unknown" class
        if potential_unknown_files:
            num_to_sample = min(len(potential_unknown_files), avg_samples)
            unknown_class_files = random.sample(potential_unknown_files, num_to_sample)

        print(f"Added 'unknown' class with {len(unknown_class_files)} samples.")
        class_names_for_training = selected_classes + ['unknown']
    else:
        class_names_for_training = selected_classes

    class_to_idx = {name: i for i, name in enumerate(class_names_for_training)}
    num_classes = len(class_names_for_training)

    # --- File and Label Preparation ---
    X = [] # file paths
    y = [] # labels

    # Add selected classes
    for class_name in selected_classes:
        class_dir = base_dir / class_name
        audio_dirs = [class_dir / 'clean'] + [class_dir / f'{snr}dB' for snr in [5, 10, 15, 20]]
        for d in audio_dirs:
            if d.exists():
                files = list(d.glob('*.wav'))
                X.extend(files)
                y.extend([class_to_idx[class_name]] * len(files))

    # Add unknown class if it exists
    if unknown_class_files:
        X.extend(unknown_class_files)
        y.extend([class_to_idx['unknown']] * len(unknown_class_files))

    # --- Train/Val/Test Split ---
    # Stratified split to maintain class distribution
    if len(np.unique(y)) < 2:
        # Cannot stratify with a single class
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    if len(np.unique(y_train_val)) < 2:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (1-test_size), random_state=42)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size / (1-test_size), random_state=42, stratify=y_train_val)


    # Return file lists for SVM
    if model_type == 'SVM':
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }, class_to_idx

    # Create PyTorch Datasets
    train_dataset = AudioDataset(X_train, y_train, target_sr, n_mels, n_fft, hop_length)
    val_dataset = AudioDataset(X_val, y_val, target_sr, n_mels, n_fft, hop_length)
    test_dataset = AudioDataset(X_test, y_test, target_sr, n_mels, n_fft, hop_length)

    # Determine if pin_memory should be used
    use_pin_memory = True
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        use_pin_memory = False # MPS does not support pinned memory

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() or 2, pin_memory=use_pin_memory),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() or 2, pin_memory=use_pin_memory),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() or 2, pin_memory=use_pin_memory)
    }

    return dataloaders, class_to_idx