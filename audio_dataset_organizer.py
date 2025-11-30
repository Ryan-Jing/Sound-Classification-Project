"""
Audio Dataset Organizer for Multi-Dataset Audio Classification
Combines UrbanSound8K, Speech Activity Detection, and Noise datasets
Preprocesses all audio to consistent format (4 seconds, 22050 Hz)
"""

import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import shutil

class AudioDatasetOrganizer:
    """
    Organizes multiple audio datasets into a unified structure using UrbanSound8K folds.
    Handles preprocessing: resampling, duration normalization, and metadata consolidation.
    """

    def __init__(self, target_sr=22050, target_duration=4.0, output_base_dir='organized_dataset'):
        """
        Initialize the dataset organizer.

        Args:
            target_sr: Target sampling rate (Hz)
            target_duration: Target duration (seconds)
            output_base_dir: Base directory for organized output
        """
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
        self.output_base_dir = Path(output_base_dir)

        # Create directory structure
        self.clean_dir = self.output_base_dir / 'clean_audio'
        self.noise_dir = self.output_base_dir / 'noise_audio'
        self.metadata_dir = self.output_base_dir / 'metadata'

        for directory in [self.clean_dir, self.noise_dir, self.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Create fold directories for clean audio (matching UrbanSound8K structure)
        for fold in range(1, 11):
            (self.clean_dir / f'fold{fold}').mkdir(exist_ok=True)

        # Master metadata
        self.metadata = {
            'clean_samples': [],
            'noise_samples': [],
            'class_mapping': {},
            'fold_distribution': {}
        }

    def process_audio_file(self, audio_path, target_path):
        """
        Load, resample, and normalize audio to target specifications.

        Args:
            audio_path: Path to input audio file
            target_path: Path to save processed audio

        Returns:
            bool: Success status
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

            # Adjust length to target duration
            if len(y) > self.target_samples:
                # Trim to target length (take from center)
                start = (len(y) - self.target_samples) // 2
                y = y[start:start + self.target_samples]
            elif len(y) < self.target_samples:
                # Pad to target length
                pad_length = self.target_samples - len(y)
                y = np.pad(y, (0, pad_length), mode='constant')

            # Normalize audio to [-1, 1] range
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))

            # Save processed audio
            sf.write(target_path, y, self.target_sr)
            return True

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return False

    def organize_urbansound8k(self, urbansound_path):
        """
        Process UrbanSound8K dataset maintaining fold structure.

        Args:
            urbansound_path: Path to UrbanSound8K dataset root
        """
        print("Processing UrbanSound8K dataset...")
        urbansound_path = Path(urbansound_path)
        metadata_file = urbansound_path / 'UrbanSound8K.csv'

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load metadata
        df = pd.read_csv(metadata_file)

        # Update class mapping
        for idx, row in df.iterrows():
            class_id = row['classID']
            class_name = row['class']
            if class_id not in self.metadata['class_mapping']:
                self.metadata['class_mapping'][class_id] = class_name

        # Process each audio file
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="UrbanSound8K"):
            fold = row['fold']
            filename = row['slice_file_name']
            class_id = row['classID']
            class_name = row['class']

            # Source and target paths
            source_path = urbansound_path / f'fold{fold}' / filename
            target_filename = f"urban_{filename}"
            target_path = self.clean_dir / f'fold{fold}' / target_filename

            # Process audio
            if source_path.exists():
                success = self.process_audio_file(source_path, target_path)

                if success:
                    # Add to metadata
                    self.metadata['clean_samples'].append({
                        'filename': target_filename,
                        'fold': fold,
                        'class_id': class_id,
                        'class_name': class_name,
                        'source_dataset': 'urbansound8k',
                        'original_filename': filename
                    })

    def organize_speech_dataset(self, speech_path, fold_assignment='balanced', exclude_noisy=True):
        """
        Process Speech Activity Detection dataset and assign to folds.

        Args:
            speech_path: Path to speech dataset root
            fold_assignment: How to assign folds ('balanced', 'sequential', or 'random')
            exclude_noisy: If True, exclude pre-noisy samples from Noizeus folder
        """
        print("Processing Speech Activity Detection dataset...")
        speech_path = Path(speech_path)

        # Find all audio files in the speech dataset
        audio_extensions = ['.wav', '.flac', '.mp3']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(speech_path.rglob(f'*{ext}')))

        # Filter out Noizeus pre-noisy samples if requested
        if exclude_noisy:
            original_count = len(audio_files)
            audio_files = [f for f in audio_files if 'Noizeus' not in str(f) and 'noizeus' not in str(f)]
            filtered_count = original_count - len(audio_files)
            if filtered_count > 0:
                print(f"  Excluded {filtered_count} pre-noisy samples from Noizeus corpus")
                print(f"  Using {len(audio_files)} clean speech samples")

        # Assign a new class ID for speech (continuing from UrbanSound8K classes)
        max_class_id = max(self.metadata['class_mapping'].keys()) if self.metadata['class_mapping'] else -1
        speech_class_id = max_class_id + 1
        self.metadata['class_mapping'][speech_class_id] = 'human_speech'

        # Distribute files across folds
        total_files = len(audio_files)
        files_per_fold = total_files // 10

        for idx, audio_file in enumerate(tqdm(audio_files, desc="Speech Dataset")):
            # Determine fold assignment
            if fold_assignment == 'balanced':
                fold = (idx % 10) + 1
            elif fold_assignment == 'sequential':
                fold = min((idx // files_per_fold) + 1, 10)
            else:  # random
                fold = np.random.randint(1, 11)

            # Generate target filename
            target_filename = f"speech_{idx:05d}.wav"
            target_path = self.clean_dir / f'fold{fold}' / target_filename

            # Process audio
            success = self.process_audio_file(audio_file, target_path)

            if success:
                # Add to metadata
                self.metadata['clean_samples'].append({
                    'filename': target_filename,
                    'fold': fold,
                    'class_id': speech_class_id,
                    'class_name': 'human_speech',
                    'source_dataset': 'speech_activity',
                    'original_filename': audio_file.name
                })

    def organize_noise_datasets(self, noise_paths):
        """
        Process noise datasets (multiple sources) into single noise pool.

        Args:
            noise_paths: List of paths to noise dataset directories
        """
        print("Processing noise datasets...")

        noise_counter = 0
        audio_extensions = ['.wav', '.flac', '.mp3', '.ogg', '.webm']

        for noise_path in noise_paths:
            noise_path = Path(noise_path)
            if not noise_path.exists():
                print(f"Warning: Noise path does not exist: {noise_path}")
                continue

            # Find all audio files
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(list(noise_path.rglob(f'*{ext}')))

            print(f"Found {len(audio_files)} noise files in {noise_path.name}")

            # Process each noise file
            for audio_file in tqdm(audio_files, desc=f"Noise - {noise_path.name}"):
                target_filename = f"noise_{noise_counter:05d}.wav"
                target_path = self.noise_dir / target_filename

                # Process audio
                success = self.process_audio_file(audio_file, target_path)

                if success:
                    # Add to metadata
                    self.metadata['noise_samples'].append({
                        'filename': target_filename,
                        'source_dataset': noise_path.name,
                        'original_filename': audio_file.name
                    })
                    noise_counter += 1

        print(f"Total noise samples processed: {noise_counter}")

    def save_metadata(self):
        """Save consolidated metadata to JSON and CSV files."""
        print("Saving metadata...")

        # Calculate fold distribution
        fold_counts = {}
        for sample in self.metadata['clean_samples']:
            fold = sample['fold']
            fold_counts[fold] = fold_counts.get(fold, 0) + 1
        self.metadata['fold_distribution'] = fold_counts

        # Save complete metadata as JSON
        with open(self.metadata_dir / 'complete_metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Save clean samples as CSV
        clean_df = pd.DataFrame(self.metadata['clean_samples'])
        clean_df.to_csv(self.metadata_dir / 'clean_samples.csv', index=False)

        # Save noise samples as CSV
        noise_df = pd.DataFrame(self.metadata['noise_samples'])
        noise_df.to_csv(self.metadata_dir / 'noise_samples.csv', index=False)

        # Save class mapping
        class_df = pd.DataFrame([
            {'class_id': k, 'class_name': v}
            for k, v in self.metadata['class_mapping'].items()
        ])
        class_df.to_csv(self.metadata_dir / 'class_mapping.csv', index=False)

        # Print summary
        print("\n=== Dataset Organization Summary ===")
        print(f"Total clean samples: {len(self.metadata['clean_samples'])}")
        print(f"Total noise samples: {len(self.metadata['noise_samples'])}")
        print(f"Number of classes: {len(self.metadata['class_mapping'])}")
        print(f"\nClass distribution:")
        for class_id, class_name in sorted(self.metadata['class_mapping'].items()):
            count = sum(1 for s in self.metadata['clean_samples'] if s['class_id'] == class_id)
            print(f"  {class_id}: {class_name} - {count} samples")
        print(f"\nFold distribution:")
        for fold, count in sorted(fold_counts.items()):
            print(f"  Fold {fold}: {count} samples")

    def validate_organized_dataset(self, num_samples_to_check=100):
        """
        Validate that organized dataset files are in correct format.

        Args:
            num_samples_to_check: Number of samples to validate per category (0 = all)
        """
        print("\n" + "=" * 70)
        print("VALIDATING ORGANIZED DATASET")
        print("=" * 70)

        errors = []

        # Validate clean audio files
        print("\nValidating clean audio files...")
        clean_files = []
        for fold in range(1, 11):
            fold_dir = self.clean_dir / f'fold{fold}'
            if fold_dir.exists():
                clean_files.extend(list(fold_dir.glob('*.wav')))

        print(f"Found {len(clean_files)} clean audio files")

        if num_samples_to_check > 0:
            import random
            clean_files = random.sample(clean_files, min(num_samples_to_check, len(clean_files)))
            print(f"Checking {len(clean_files)} random samples...")

        for audio_file in tqdm(clean_files, desc="Validating clean audio"):
            try:
                y, sr = librosa.load(audio_file, sr=None, mono=True)
                if sr != self.target_sr:
                    errors.append(f"{audio_file.name}: Wrong sample rate {sr} (expected {self.target_sr})")
                if len(y) != self.target_samples:
                    errors.append(f"{audio_file.name}: Wrong length {len(y)} (expected {self.target_samples})")
                if np.max(np.abs(y)) > 1.01:  # Allow small margin for floating point
                    errors.append(f"{audio_file.name}: Not normalized, max = {np.max(np.abs(y))}")
            except Exception as e:
                errors.append(f"{audio_file.name}: Load error - {str(e)}")

        # Validate noise audio files
        print("\nValidating noise audio files...")
        noise_files = list(self.noise_dir.glob('*.wav'))
        print(f"Found {len(noise_files)} noise audio files")

        if num_samples_to_check > 0:
            noise_files = random.sample(noise_files, min(num_samples_to_check, len(noise_files)))
            print(f"Checking {len(noise_files)} random samples...")

        for audio_file in tqdm(noise_files, desc="Validating noise audio"):
            try:
                y, sr = librosa.load(audio_file, sr=None, mono=True)
                if sr != self.target_sr:
                    errors.append(f"{audio_file.name}: Wrong sample rate {sr} (expected {self.target_sr})")
                if len(y) != self.target_samples:
                    errors.append(f"{audio_file.name}: Wrong length {len(y)} (expected {self.target_samples})")
                if np.max(np.abs(y)) > 1.01:
                    errors.append(f"{audio_file.name}: Not normalized, max = {np.max(np.abs(y))}")
            except Exception as e:
                errors.append(f"{audio_file.name}: Load error - {str(e)}")

        # Report results
        print("\n" + "=" * 70)
        if len(errors) == 0:
            print("✅ VALIDATION PASSED - All checked files are correctly formatted!")
        else:
            print(f"⚠️  VALIDATION FOUND {len(errors)} ISSUES:")
            for error in errors[:20]:  # Show first 20 errors
                print(f"  - {error}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more errors")
        print("=" * 70)


def main():
    """
    Main execution function.
    Update the paths below to match your Kaggle dataset locations.
    """

    # Initialize organizer
    organizer = AudioDatasetOrganizer(
        target_sr=22050,
        target_duration=4.0,
        output_base_dir='organized_dataset'
    )

    # Dataset paths - CORRECTED to match actual folder structure
    URBANSOUND8K_PATH = 'datasets/urbansound8k'
    SPEECH_DATASET_PATH = 'datasets/speech-activity-detection-datasets/Audio'  # Point to Audio subfolder
    NOISE_DATASET_PATHS = [
        'datasets/audio-noise-dataset',      # Contains .webm files
        'datasets/noise-data-set'            # Contains clean_train, noise_train folders
    ]

    print("=" * 70)
    print("AUDIO DATASET ORGANIZATION PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Target sample rate: {organizer.target_sr} Hz")
    print(f"  Target duration: {organizer.target_duration} seconds")
    print(f"  Output directory: {organizer.output_base_dir}")
    print(f"\nDataset sources:")
    print(f"  1. UrbanSound8K: {URBANSOUND8K_PATH}")
    print(f"  2. Speech Audio: {SPEECH_DATASET_PATH}")
    print(f"  3. Noise datasets: {len(NOISE_DATASET_PATHS)} sources")
    print("=" * 70)

    # Process datasets
    print("\n[STEP 1/5] Processing UrbanSound8K...")
    organizer.organize_urbansound8k(URBANSOUND8K_PATH)

    print("\n[STEP 2/5] Processing Speech Dataset (excluding pre-noisy Noizeus samples)...")
    organizer.organize_speech_dataset(SPEECH_DATASET_PATH, fold_assignment='balanced', exclude_noisy=True)

    print("\n[STEP 3/5] Processing Noise Datasets...")
    organizer.organize_noise_datasets(NOISE_DATASET_PATHS)

    print("\n[STEP 4/5] Saving Metadata...")
    organizer.save_metadata()

    print("\n[STEP 5/5] Validating Organized Dataset...")
    organizer.validate_organized_dataset(num_samples_to_check=100)

    print("\n" + "=" * 70)
    print("✅ DATASET ORGANIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput directory: {organizer.output_base_dir}")
    print(f"\nYou can now run training with:")
    print(f"  python training_pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
