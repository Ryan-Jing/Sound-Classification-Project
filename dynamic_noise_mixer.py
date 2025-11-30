"""
Dynamic Audio Mixer for Training
Mixes clean audio with noise on-the-fly during training without saving intermediate files.
Supports progressive noise curriculum learning with increasing noise levels.
"""

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import random


class DynamicNoiseMixer:
    """
    Handles dynamic mixing of clean audio with noise samples.
    Supports SNR-based mixing and progressive noise curriculum.
    """
    
    def __init__(self, noise_dir, target_sr=22050):
        """
        Initialize the noise mixer.
        
        Args:
            noise_dir: Directory containing preprocessed noise files
            target_sr: Sampling rate (should match preprocessed audio)
        """
        self.noise_dir = Path(noise_dir)
        self.target_sr = target_sr
        
        # Load all noise file paths
        self.noise_files = list(self.noise_dir.glob('*.wav'))
        if len(self.noise_files) == 0:
            raise ValueError(f"No noise files found in {noise_dir}")
        
        print(f"Loaded {len(self.noise_files)} noise samples")
        
        # Cache for loaded noise samples (optional, for memory efficiency)
        self.noise_cache = {}
        
    def load_noise_sample(self, noise_idx=None):
        """
        Load a random noise sample or specific noise by index.
        
        Args:
            noise_idx: Optional specific noise index, otherwise random
            
        Returns:
            numpy array: Noise audio signal
        """
        if noise_idx is None:
            noise_idx = random.randint(0, len(self.noise_files) - 1)
        
        # Check cache first
        if noise_idx in self.noise_cache:
            return self.noise_cache[noise_idx].copy()
        
        # Load noise file
        noise_path = self.noise_files[noise_idx]
        noise, _ = librosa.load(noise_path, sr=self.target_sr, mono=True)

        # Optionally cache (be careful with memory)
        # Increased to 500 for better performance with large noise datasets
        # Each 4-second audio at 22050 Hz is ~88KB, so 500 samples ≈ 44MB
        if len(self.noise_cache) < 500:  # Cache first 500 noise samples
            self.noise_cache[noise_idx] = noise.copy()

        return noise
    
    def mix_audio_snr(self, clean_audio, noise_audio, snr_db):
        """
        Mix clean audio with noise at specified Signal-to-Noise Ratio.
        
        Args:
            clean_audio: Clean audio signal (numpy array)
            noise_audio: Noise audio signal (numpy array)
            snr_db: Target SNR in decibels
            
        Returns:
            numpy array: Mixed audio signal
        """
        # Calculate signal and noise power
        signal_power = np.mean(clean_audio ** 2)
        noise_power = np.mean(noise_audio ** 2)
        
        # Calculate required noise scaling factor for target SNR
        # SNR = 10 * log10(signal_power / noise_power)
        # noise_power_target = signal_power / (10 ^ (SNR/10))
        snr_linear = 10 ** (snr_db / 10)
        noise_power_target = signal_power / snr_linear
        
        # Scale noise to achieve target SNR
        if noise_power > 0:
            noise_scaling = np.sqrt(noise_power_target / noise_power)
        else:
            noise_scaling = 0
        
        scaled_noise = noise_audio * noise_scaling
        
        # Mix signals
        mixed_audio = clean_audio + scaled_noise
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val
        
        return mixed_audio
    
    def mix_audio_percentage(self, clean_audio, noise_audio, noise_percentage):
        """
        Mix clean audio with noise at specified percentage (0-100).
        
        Args:
            clean_audio: Clean audio signal (numpy array)
            noise_audio: Noise audio signal (numpy array)
            noise_percentage: Noise level as percentage (0-100)
            
        Returns:
            numpy array: Mixed audio signal
        """
        # Convert percentage to mixing ratio
        clean_weight = 1.0
        noise_weight = noise_percentage / 100.0
        
        # Mix signals
        mixed_audio = clean_audio * clean_weight + noise_audio * noise_weight
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 1.0:
            mixed_audio = mixed_audio / max_val
        
        return mixed_audio
    
    def get_mixed_audio(self, clean_audio, snr_db=None, noise_percentage=None):
        """
        Get mixed audio with randomly selected noise.
        
        Args:
            clean_audio: Clean audio signal (numpy array)
            snr_db: Target SNR in dB (if using SNR-based mixing)
            noise_percentage: Noise percentage 0-100 (if using percentage-based mixing)
            
        Returns:
            numpy array: Mixed audio signal
        """
        # Load random noise sample
        noise = self.load_noise_sample()
        
        # Match noise length to clean audio length
        if len(noise) > len(clean_audio):
            # Random crop from noise
            start_idx = random.randint(0, len(noise) - len(clean_audio))
            noise = noise[start_idx:start_idx + len(clean_audio)]
        elif len(noise) < len(clean_audio):
            # Tile noise to match length
            repeats = int(np.ceil(len(clean_audio) / len(noise)))
            noise = np.tile(noise, repeats)[:len(clean_audio)]
        
        # Mix based on specified method
        if snr_db is not None:
            return self.mix_audio_snr(clean_audio, noise, snr_db)
        elif noise_percentage is not None:
            return self.mix_audio_percentage(clean_audio, noise, noise_percentage)
        else:
            raise ValueError("Must specify either snr_db or noise_percentage")


class AudioDatasetWithDynamicNoise(Dataset):
    """
    PyTorch Dataset that loads clean audio and applies dynamic noise mixing.
    Supports curriculum learning with progressive noise levels.
    """
    
    def __init__(self, metadata_csv, clean_audio_dir, noise_mixer, 
                 fold=1, mode='train', transform=None, 
                 noise_curriculum=None, current_epoch=0):
        """
        Initialize dataset.
        
        Args:
            metadata_csv: Path to clean samples metadata CSV
            clean_audio_dir: Directory containing clean audio files
            noise_mixer: DynamicNoiseMixer instance
            fold: Which fold to use (1-10)
            mode: 'train' or 'test' (test uses different folds)
            transform: Optional feature extraction transform
            noise_curriculum: Dict defining noise progression (e.g., {'epochs': [0, 10, 20], 'snr_db': [inf, 20, 10]})
            current_epoch: Current training epoch for curriculum learning
        """
        self.clean_audio_dir = Path(clean_audio_dir)
        self.noise_mixer = noise_mixer
        self.transform = transform
        self.mode = mode
        self.current_epoch = current_epoch
        self.noise_curriculum = noise_curriculum
        
        # Load metadata
        df = pd.read_csv(metadata_csv)
        
        # Filter by fold (train uses all except test fold, test uses only test fold)
        if mode == 'train':
            self.samples = df[df['fold'] != fold].reset_index(drop=True)
        else:  # test
            self.samples = df[df['fold'] == fold].reset_index(drop=True)
        
        print(f"Loaded {len(self.samples)} samples for {mode} (fold {fold})")
        
    def set_epoch(self, epoch):
        """Update current epoch for curriculum learning."""
        self.current_epoch = epoch
    
    def get_noise_level(self):
        """
        Determine current noise level based on curriculum and epoch.
        
        Returns:
            float or None: SNR in dB, or None for clean audio
        """
        if self.noise_curriculum is None:
            return None  # No noise
        
        epochs = self.noise_curriculum.get('epochs', [0])
        snr_levels = self.noise_curriculum.get('snr_db', [float('inf')])
        
        # Find appropriate noise level for current epoch
        for i in range(len(epochs) - 1, -1, -1):
            if self.current_epoch >= epochs[i]:
                snr = snr_levels[i]
                return None if snr == float('inf') else snr
        
        return None  # Default to clean
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample with dynamic noise mixing.
        
        Returns:
            tuple: (features, label) where features are MFCC or raw audio
        """
        # Get sample metadata
        sample = self.samples.iloc[idx]
        fold = sample['fold']
        filename = sample['filename']
        label = sample['class_id']
        
        # Load clean audio
        audio_path = self.clean_audio_dir / f'fold{fold}' / filename
        clean_audio, _ = librosa.load(audio_path, sr=self.noise_mixer.target_sr, mono=True)
        
        # Apply noise based on curriculum
        snr_db = self.get_noise_level()
        
        if snr_db is None:
            # Use clean audio
            audio = clean_audio
        else:
            # Mix with noise at current SNR level
            audio = self.noise_mixer.get_mixed_audio(clean_audio, snr_db=snr_db)
        
        # Apply feature extraction transform if provided
        if self.transform is not None:
            features = self.transform(audio)
        else:
            # Return raw audio as tensor
            features = torch.FloatTensor(audio)
        
        return features, label


class MFCCTransform:
    """Transform to extract MFCC features from audio signal."""
    
    def __init__(self, sr=22050, n_mfcc=40, n_fft=2048, hop_length=512):
        """
        Initialize MFCC transform.
        
        Args:
            sr: Sampling rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def __call__(self, audio):
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio signal (numpy array)
            
        Returns:
            torch.Tensor: MFCC features as 2D tensor
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sr, 
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to tensor
        return torch.FloatTensor(mfccs)


def create_dataloaders(organized_dataset_dir, test_fold=1, batch_size=32, 
                       noise_curriculum=None, num_workers=4):
    """
    Create train and test dataloaders with dynamic noise mixing.
    
    Args:
        organized_dataset_dir: Path to organized dataset directory
        test_fold: Which fold to use for testing (1-10)
        batch_size: Batch size for dataloaders
        noise_curriculum: Curriculum for progressive noise (e.g., {'epochs': [0, 10, 20, 30], 'snr_db': [inf, 20, 15, 10]})
        num_workers: Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader, noise_mixer)
    """
    organized_dataset_dir = Path(organized_dataset_dir)
    
    # Initialize noise mixer
    noise_mixer = DynamicNoiseMixer(
        noise_dir=organized_dataset_dir / 'noise_audio',
        target_sr=22050
    )
    
    # Initialize MFCC transform
    mfcc_transform = MFCCTransform(sr=22050, n_mfcc=40)
    
    # Create datasets
    train_dataset = AudioDatasetWithDynamicNoise(
        metadata_csv=organized_dataset_dir / 'metadata' / 'clean_samples.csv',
        clean_audio_dir=organized_dataset_dir / 'clean_audio',
        noise_mixer=noise_mixer,
        fold=test_fold,
        mode='train',
        transform=mfcc_transform,
        noise_curriculum=noise_curriculum,
        current_epoch=0
    )
    
    test_dataset = AudioDatasetWithDynamicNoise(
        metadata_csv=organized_dataset_dir / 'metadata' / 'clean_samples.csv',
        clean_audio_dir=organized_dataset_dir / 'clean_audio',
        noise_mixer=noise_mixer,
        fold=test_fold,
        mode='test',
        transform=mfcc_transform,
        noise_curriculum=None,  # Always use clean audio for testing
        current_epoch=0
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    
    return train_loader, test_loader, train_dataset


# Example usage
if __name__ == "__main__":
    # Define noise curriculum
    # Start with clean audio, progressively add noise over epochs
    noise_curriculum = {
        'epochs': [0, 5, 10, 15, 20, 25],
        'snr_db': [float('inf'), 25, 20, 15, 12, 10]  # inf means clean audio
    }
    
    # Create dataloaders
    train_loader, test_loader, train_dataset = create_dataloaders(
        organized_dataset_dir='organized_dataset',
        test_fold=1,
        batch_size=32,
        noise_curriculum=noise_curriculum,
        num_workers=4
    )
    
    # Example: Training loop with curriculum learning
    print("\nExample training loop structure:")
    print("=" * 50)
    
    for epoch in range(30):
        # Update dataset epoch for curriculum learning
        train_dataset.set_epoch(epoch)
        
        # Get current noise level for logging
        snr = train_dataset.get_noise_level()
        if snr is None:
            print(f"Epoch {epoch}: Training with CLEAN audio")
        else:
            print(f"Epoch {epoch}: Training with SNR = {snr} dB")
        
        # Your training code here
        for batch_idx, (features, labels) in enumerate(train_loader):
            # features shape: [batch_size, n_mfcc, time_frames]
            # labels shape: [batch_size]
            
            if batch_idx == 0:  # Just show first batch info
                print(f"  Batch features shape: {features.shape}")
                print(f"  Batch labels shape: {labels.shape}")
            
            # Your model training code would go here
            # loss = criterion(model(features), labels)
            # loss.backward()
            # optimizer.step()
            
            break  # Just showing first batch for example
    
    print("\nDataloader creation successful!")
