import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random
import warnings

def calculate_rms(audio):
    """Calculate the Root Mean Square of an audio signal."""
    return np.sqrt(np.mean(audio**2))

def adjust_noise_level(clean_rms, noise, snr_db):
    """Adjust noise to a specific SNR relative to clean audio."""
    noise_rms = calculate_rms(noise)
    if noise_rms == 0:
        return noise  # Cannot adjust if noise is silent

    target_noise_rms = clean_rms / (10**(snr_db / 20))

    # Avoid division by zero
    if noise_rms > 1e-6:
        noise = noise * (target_noise_rms / noise_rms)
    return noise

def mix_audio(clean_audio, noise_audio, snr_db):
    """Mix clean audio with noise at a specified SNR."""
    # Ensure audio signals are numpy arrays
    clean_audio = np.array(clean_audio)
    noise_audio = np.array(noise_audio)

    # Ensure noise is long enough, loop if necessary
    if len(noise_audio) < len(clean_audio):
        repeats = int(np.ceil(len(clean_audio) / len(noise_audio)))
        noise_audio = np.tile(noise_audio, repeats)

    # Trim noise to match clean audio length
    noise_audio = noise_audio[:len(clean_audio)]

    clean_rms = calculate_rms(clean_audio)

    # Adjust noise to desired SNR
    adjusted_noise = adjust_noise_level(clean_rms, noise_audio, snr_db)

    # Mix audio
    mixed_audio = clean_audio + adjusted_noise

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed_audio))
    if max_val > 1.0:
        mixed_audio /= max_val

    return mixed_audio

class NewAudioDatasetOrganizer:
    def __init__(self, target_sr=16000, target_duration=3.0, output_base_dir='data/organized_audio_datasets'):
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.target_samples = int(target_sr * target_duration)
        self.output_base_dir = Path(output_base_dir)
        self.snr_levels_db = [5, 10, 15, 20]

        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created at: {self.output_base_dir.resolve()}")

    def process_and_save(self, audio_path, target_path):
        """Loads, resamples, and normalizes a single audio file."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)

            # Ignore clips shorter than the target duration
            if len(y) < self.target_samples:
                return None

            # Truncate longer clips from the center
            if len(y) > self.target_samples:
                start = (len(y) - self.target_samples) // 2
                y = y[start:start + self.target_samples]

            # Normalize audio
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y /= max_val

            sf.write(target_path, y, self.target_sr)
            return y, self.target_sr
        except Exception as e:
            # print(f"Skipping {audio_path}: {e}")
            return None

    def run(self):
        # --- 1. Identify all audio files ---
        print("Step 1: Identifying all audio files...")

        # Noise files
        noise_paths = []
        noise_data_set_path = Path('data/raw_datasets/noise-data-set')
        noise_paths.extend(list((noise_data_set_path / 'noise_train').rglob('*.wav')))
        noise_paths.extend(list((noise_data_set_path / 'noise_test').rglob('*.wav')))

        urbansound_path = Path('data/raw_datasets/urbansound8k')
        urbansound_df = pd.read_csv(urbansound_path / 'UrbanSound8K.csv')
        air_conditioner_files = urbansound_df[urbansound_df['class'] == 'air_conditioner']
        for _, row in air_conditioner_files.iterrows():
            noise_paths.append(urbansound_path / f"fold{row['fold']}" / row['slice_file_name'])

        print(f"Found {len(noise_paths)} total noise files.")

        # Feature files
        feature_map = {}

        # UrbanSound8K features
        us8k_features = urbansound_df[urbansound_df['class'] != 'air_conditioner']
        for _, row in us8k_features.iterrows():
            class_name = row['class']
            if class_name not in feature_map:
                feature_map[class_name] = []
            feature_map[class_name].append(urbansound_path / f"fold{row['fold']}" / row['slice_file_name'])

        # Human speech features
        speech_paths = []
        speech_paths.extend(list(Path('data/raw_datasets/live-speech-dataset/wavs').rglob('*.wav')))
        speech_paths.extend(list((noise_data_set_path / 'clean_train').rglob('*.wav')))
        speech_paths.extend(list((noise_data_set_path / 'clean_test').rglob('*.wav')))
        speech_paths.extend(list(Path('data/raw_datasets/speech-activity-detection-datasets/Audio/Female').rglob('*.wav')))
        speech_paths.extend(list(Path('data/raw_datasets/speech-activity-detection-datasets/Audio/Male').rglob('*.wav')))
        feature_map['human_speech'] = speech_paths

        print(f"Found {len(feature_map.keys())} feature classes, including human_speech.")

        # --- 2. Process and create augmented datasets ---
        print("\nStep 2: Processing features and augmenting with noise...")
        for class_name, files in feature_map.items():
            print(f"\nProcessing class: {class_name}")

            # Create directories for the class
            class_dir = self.output_base_dir / class_name
            clean_dir = class_dir / 'clean'
            clean_dir.mkdir(parents=True, exist_ok=True)
            for snr in self.snr_levels_db:
                (class_dir / f'{snr}dB').mkdir(exist_ok=True)

            for audio_file_path in tqdm(files, desc=f"  {class_name}"):
                base_filename = audio_file_path.stem

                # Process and save the clean version
                clean_target_path = clean_dir / f"{base_filename}_clean.wav"
                processed_result = self.process_and_save(audio_file_path, clean_target_path)

                if processed_result:
                    clean_audio, sr = processed_result

                    # Create noisy versions
                    for snr in self.snr_levels_db:
                        noise_file_path = random.choice(noise_paths)

                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                noise_audio, _ = librosa.load(noise_file_path, sr=self.target_sr, mono=True)
                        except Exception as e:
                            # print(f"Could not load noise file {noise_file_path}, skipping: {e}")
                            continue

                        mixed_audio = mix_audio(clean_audio, noise_audio, snr)

                        noisy_target_path = class_dir / f'{snr}dB' / f"{base_filename}_snr{snr}.wav"
                        sf.write(noisy_target_path, mixed_audio, self.target_sr)

        # --- 3. Process Noizeus validation set ---
        print("\nStep 3: Processing Noizeus validation set...")
        noizeus_path = Path('data/raw_datasets/speech-activity-detection-datasets/Audio/Noizeus')
        noizeus_validation_dir = self.output_base_dir / 'human_speech' / 'validation_noizeus'
        noizeus_validation_dir.mkdir(exist_ok=True)

        noizeus_files = list(noizeus_path.rglob('*.wav'))
        for audio_file_path in tqdm(noizeus_files, desc="  Noizeus"):
            target_path = noizeus_validation_dir / audio_file_path.name
            self.process_and_save(audio_file_path, target_path)

        print("\nOrganization complete!")

if __name__ == '__main__':
    organizer = NewAudioDatasetOrganizer()
    organizer.run()
