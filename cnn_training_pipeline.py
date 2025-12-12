import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose
import librosa
import numpy as np
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import datetime
import pandas as pd
from pathlib import Path
import random

# ============================================================================
# Feature Extraction
# ============================================================================

class MFCCTransform:
    """Transform to extract MFCC features from audio signal."""

    def __init__(self, sr=22050, n_mfcc=40, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, audio):
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return torch.FloatTensor(mfccs)

# ============================================================================
# Model Definition
# ============================================================================

class SpeechCNN(nn.Module):
    """CNN model for speech activity detection from MFCC features."""

    def __init__(self, num_classes=2, n_mfcc=40):
        super(SpeechCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # This will be set dynamically based on input size in the first forward pass
        self.fc1 = None
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: [batch, n_mfcc, time_frames]
        # Add channel dimension
        x = x.unsqueeze(1)  # [batch, 1, n_mfcc, time_frames]

        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)

        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

# ============================================================================
# Dynamic Noise Mixer
# ============================================================================

class DynamicNoiseMixer:
    """
    Handles dynamic mixing of clean audio with noise samples.
    Supports SNR-based mixing and progressive noise curriculum.
    """

    def __init__(self, noise_dir, target_sr=22050):
        self.noise_dir = Path(noise_dir)
        self.target_sr = target_sr

        self.noise_files = list(self.noise_dir.glob('*.wav'))
        if len(self.noise_files) == 0:
            print(f"Warning: No noise files found in {noise_dir}. Noise mixing will be skipped.")

        self.noise_cache = {}

    def load_noise_sample(self, noise_idx=None):
        if len(self.noise_files) == 0:
            return None # No noise files to load

        if noise_idx is None:
            noise_idx = random.randint(0, len(self.noise_files) - 1)

        if noise_idx in self.noise_cache:
            return self.noise_cache[noise_idx].copy()

        noise_path = self.noise_files[noise_idx]
        noise, _ = librosa.load(noise_path, sr=self.target_sr, mono=True)

        if len(self.noise_cache) < 500:
            self.noise_cache[noise_idx] = noise.copy()

        return noise

    def mix_audio_snr(self, clean_audio, noise_audio, snr_db):
        if noise_audio is None:
            return clean_audio # No noise to mix

        signal_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise_audio ** 2)

        snr_linear = 10 ** (snr_db / 10)
        noise_power_target = signal_power / snr_linear

        if noise_power > 0:
            noise_scaling = np.sqrt(noise_power_target / noise_power)
        else:
            noise_scaling = 0

        scaled_noise = noise_audio * noise_scaling
        mixed_audio = clean_audio + scaled_noise

        max_val = np.max(np.abs(mixed_audio))
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val

        return mixed_audio

    def get_mixed_audio(self, clean_audio, snr_db=None):
        if snr_db is None or snr_db == float('inf'):
            return clean_audio

        noise = self.load_noise_sample()
        if noise is None:
            return clean_audio # No noise files, return clean audio

        if len(noise) > len(clean_audio):
            start_idx = random.randint(0, len(noise) - len(clean_audio))
            noise = noise[start_idx:start_idx + len(clean_audio)]
        elif len(noise) < len(clean_audio):
            repeats = int(np.ceil(len(clean_audio) / len(noise)))
            noise = np.tile(noise, repeats)[:len(clean_audio)]

        return self.mix_audio_snr(clean_audio, noise, snr_db)

# ============================================================================
# Dataset
# ============================================================================

class SpeechDataset(Dataset):
    """
    PyTorch Dataset that loads speech/no-speech audio and applies dynamic noise mixing.
    """

    def __init__(self, metadata_csv, data_base_dir, noise_mixer, transform=None,
                 noise_curriculum=None, current_epoch=0, target_sr=22050):
        self.data_base_dir = Path(data_base_dir)
        self.noise_mixer = noise_mixer
        self.transform = transform
        self.current_epoch = current_epoch
        self.noise_curriculum = noise_curriculum
        self.target_sr = target_sr

        self.samples = pd.read_csv(metadata_csv)

        print(f"Loaded {len(self.samples)} samples from {metadata_csv}")

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def get_noise_level(self):
        if self.noise_curriculum is None:
            return None

        epochs = self.noise_curriculum.get('epochs', [0])
        snr_levels = self.noise_curriculum.get('snr_db', [float('inf')])

        for i in range(len(epochs) - 1, -1, -1):
            if self.current_epoch >= epochs[i]:
                snr = snr_levels[i]
                return None if snr == float('inf') else snr

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx]
        filepath = self.data_base_dir / sample['file']
        label = sample['label']

        clean_audio, _ = librosa.load(filepath, sr=self.target_sr, mono=True)

        snr_db = self.get_noise_level()
        audio = self.noise_mixer.get_mixed_audio(clean_audio, snr_db=snr_db)

        if self.transform is not None:
            features = self.transform(audio)
        else:
            features = torch.FloatTensor(audio)

        return features, label

# ============================================================================
# Data Loaders Creation
# ============================================================================

def create_speech_dataloaders(organized_data_dir, batch_size=32,
                              noise_curriculum=None, num_workers=4, target_sr=22050):

    organized_data_dir = Path(organized_data_dir)

    noise_mixer = DynamicNoiseMixer(
        noise_dir=organized_data_dir / 'training_noise', # Corrected path
        target_sr=target_sr
    )

    mfcc_transform = MFCCTransform(sr=target_sr, n_mfcc=40)

    train_dataset = SpeechDataset(
        metadata_csv=organized_data_dir / 'train.csv',
        data_base_dir=organized_data_dir,
        noise_mixer=noise_mixer,
        transform=mfcc_transform,
        noise_curriculum=noise_curriculum,
        target_sr=target_sr
    )

    val_dataset = SpeechDataset(
        metadata_csv=organized_data_dir / 'val.csv',
        data_base_dir=organized_data_dir,
        noise_mixer=noise_mixer, # Val and test also get noise mixer for consistency but snr should be inf
        transform=mfcc_transform,
        noise_curriculum={'epochs': [0], 'snr_db': [float('inf')]}, # Always clean for validation
        target_sr=target_sr
    )

    test_dataset = SpeechDataset(
        metadata_csv=organized_data_dir / 'test.csv',
        data_base_dir=organized_data_dir,
        noise_mixer=noise_mixer,
        transform=mfcc_transform,
        noise_curriculum={'epochs': [0], 'snr_db': [float('inf')]}, # Always clean for test
        target_sr=target_sr
    )

    # Create an additional dataset for pure noise testing
    pure_noise_test_dataset = PureNoiseDataset(
        noise_dir=organized_data_dir / 'test_noise',
        transform=mfcc_transform,
        target_sr=target_sr
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    pure_noise_test_loader = DataLoader(
        pure_noise_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, pure_noise_test_loader, train_dataset


class PureNoiseDataset(Dataset):
    """
    Dataset for loading pure noise audio clips for testing.
    All labels are 0 (no-speech).
    """
    def __init__(self, noise_dir, transform=None, target_sr=22050):
        self.noise_dir = Path(noise_dir)
        self.transform = transform
        self.target_sr = target_sr

        self.noise_files = list(self.noise_dir.glob('*.wav'))

        print(f"Loaded {len(self.noise_files)} pure noise samples from {noise_dir}")

    def __len__(self):
        return len(self.noise_files)

    def __getitem__(self, idx):
        filepath = self.noise_files[idx]

        audio, _ = librosa.load(filepath, sr=self.target_sr, mono=True)

        if self.transform is not None:
            features = self.transform(audio)
        else:
            features = torch.FloatTensor(audio)

        # Label for pure noise is always 0 (no-speech)
        label = 0

        return features, label

# ============================================================================
# Training Functions
# ============================================================================

class CNNTrainer:
    """Trainer for CNN model with curriculum learning."""

    def __init__(self, model, device, num_classes=2):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for features, labels in pbar:
            features, labels = features.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })

        return total_loss / len(train_loader), 100. * correct / total

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100. * correct / total
        return accuracy, all_preds, all_labels


def train_speech_cnn(organized_data_dir='speech_data_organized',
                     num_epochs=50, batch_size=32,
                     learning_rate=0.001, device='auto',
                     model_save_name='best_speech_cnn.pth'):
    """
    Complete training pipeline for Speech CNN with progressive noise curriculum.
    """

    if device == 'auto':
        if torch.cuda.is_available():
            selected_device = 'cuda'
        elif torch.backends.mps.is_available():
            selected_device = 'mps'
        else:
            selected_device = 'cpu'
    else:
        selected_device = device

    device = torch.device(selected_device)
    print(f"Using device: {device}")

    num_classes = 2 # Speech (1) or No-Speech (0)

    # Define noise curriculum (simplified to two stages)
    # Stage 1: Clean audio (first half of epochs)
    # Stage 2: Noisy audio with a fixed SNR (second half of epochs)
    clean_epochs = num_epochs // 2
    noisy_epochs = num_epochs - clean_epochs

    noise_curriculum = {
        'epochs': [0, clean_epochs],
        'snr_db': [float('inf'), 15] # Clean for first half, 15 dB SNR for second half
    }

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, pure_noise_test_loader, train_dataset = create_speech_dataloaders(
        organized_data_dir=organized_data_dir,
        batch_size=batch_size,
        noise_curriculum=noise_curriculum,
        num_workers=4
    )

    print("\nInitializing SpeechCNN model...")
    model = SpeechCNN(num_classes=num_classes, n_mfcc=40)
    trainer = CNNTrainer(model, device, num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler to reduce LR for noisy phase
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[clean_epochs], gamma=0.1)

    results = {
        'model_type': 'SpeechCNN',
        'epochs': [],
        'final_test_accuracy': None,
        'final_pure_noise_accuracy': None
    }

    print("\n" + "=" * 70)
    print("Starting SpeechCNN training with noise curriculum")
    print("=" * 70)

    best_val_acc = 0
    best_epoch = -1

    for epoch in range(num_epochs):
        train_dataset.set_epoch(epoch)
        snr = train_dataset.get_noise_level()

        if snr is None:
            print(f"\nEpoch {epoch+1}/{num_epochs} - Training with CLEAN audio (LR: {optimizer.param_groups[0]['lr']:.6f})")
        else:
            print(f"\nEpoch {epoch+1}/{num_epochs} - Training with SNR = {snr} dB (LR: {optimizer.param_groups[0]['lr']:.6f})")

        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, epoch)
        val_acc, _, _ = trainer.evaluate(val_loader)

        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Validation Acc: {val_acc:.2f}%")

        # Save results for plotting
        results['epochs'].append({
            'epoch': epoch,
            'snr_db': snr,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc # Added for consistency with visualize_results.py
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_name)
            print(f"  New best model saved! (Validation Acc: {best_val_acc:.2f}%)")

        scheduler.step() # Step the learning rate scheduler

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}")

    # Final evaluation on test set
    print("\nLoading best model for final evaluation on test set...")
    model.load_state_dict(torch.load(model_save_name))
    test_acc, test_preds, test_labels = trainer.evaluate(test_loader)
    print(f"  Final Test Accuracy (Speech/No-Speech): {test_acc:.2f}%")
    results['final_test_accuracy'] = test_acc

    # Evaluate on pure noise test set
    pure_noise_acc, pure_noise_preds, pure_noise_labels = trainer.evaluate(pure_noise_test_loader)
    print(f"  Final Test Accuracy (Pure Noise): {pure_noise_acc:.2f}%")
    results['final_pure_noise_accuracy'] = pure_noise_acc

    # Save final results
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = Path('results') / f'speechcnn_fold1_{timestamp}' / f'speechcnn_results_fold1.json'
    results_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print(f"Best model saved to {model_save_name}")

    # Save predictions and labels for visualization
    np.save(results_path.parent / 'test_preds.npy', np.array(test_preds))
    np.save(results_path.parent / 'test_labels.npy', np.array(test_labels))
    np.save(results_path.parent / 'pure_noise_preds.npy', np.array(pure_noise_preds))
    np.save(results_path.parent / 'pure_noise_labels.npy', np.array(pure_noise_labels))
    print(f"Predictions and labels saved to {results_path.parent}/")

    return results, test_preds, test_labels, pure_noise_preds, pure_noise_labels


if __name__ == '__main__':
    results, test_preds, test_labels, pure_noise_preds, pure_noise_labels = train_speech_cnn(
        organized_data_dir='data/speech_data_organized',
        num_epochs=10, # Reduced for faster example
        learning_rate=0.001
    )