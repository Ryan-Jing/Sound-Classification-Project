
import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re
from sklearn.model_selection import train_test_split

TARGET_SR = 22050
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
OUTPUT_BASE_DIR = Path('data/speech_data_organized')

def process_audio_file(audio, target_sr, target_samples):
    """
    Resample and normalize audio to target specifications.
    """
    if len(audio) > target_samples:
        start = (len(audio) - target_samples) // 2
        audio = audio[start:start + target_samples]
    elif len(audio) < target_samples:
        pad_length = target_samples - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio

def parse_textgrid(textgrid_path):
    """
    Parses a TextGrid file to extract intervals and their labels.
    """
    with open(textgrid_path, 'r') as f:
        content = f.read()

    intervals = []

    # This is a simplified parser based on the example format.
    # It might need to be more robust for different TextGrid variations.

    text = content.split('item [1]:')[1] # Assumes the interesting data is in the first item.

    intervals_raw = text.split('intervals [')[1:]
    for interval_raw in intervals_raw:
        lines = interval_raw.splitlines()
        if len(lines) < 4:
            continue

        xmin_line = lines[1]
        xmax_line = lines[2]
        text_line = lines[3]

        try:
            xmin = float(re.search(r'xmin = ([\d.]+)', xmin_line).group(1))
            xmax = float(re.search(r'xmax = ([\d.]+)', xmax_line).group(1))
            label = re.search(r'text = "(\d)"', text_line).group(1)
            intervals.append({'xmin': xmin, 'xmax': xmax, 'label': label})
        except (AttributeError, IndexError):
            # Could not parse this interval, skip it
            continue

    return intervals


def organize_speech_data():
    """
    Organizes the speech activity detection dataset into speech and no-speech clips.
    """
    speech_dir = OUTPUT_BASE_DIR / 'speech'
    no_speech_dir = OUTPUT_BASE_DIR / 'no_speech'
    test_noise_dir = OUTPUT_BASE_DIR / 'test_noise'
    training_noise_dir = OUTPUT_BASE_DIR / 'training_noise' # New directory for general training noise

    for directory in [speech_dir, no_speech_dir, test_noise_dir, training_noise_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    annotation_base_path = Path('data/raw_datasets/speech-activity-detection-datasets/Annotation')
    audio_base_path = Path('data/raw_datasets/speech-activity-detection-datasets/Audio')

    textgrid_files = list(annotation_base_path.glob('**/*.TextGrid'))

    all_files = []

    speech_count = 0
    no_speech_count = 0

    for textgrid_file in tqdm(textgrid_files, desc="Processing TextGrids"):
        if 'Noizeus' in str(textgrid_file):
            continue

        relative_path = textgrid_file.relative_to(annotation_base_path)
        audio_file_path_wav = (audio_base_path / relative_path).with_suffix('.wav')
        audio_file_path_flac = (audio_base_path / relative_path).with_suffix('.flac')

        if audio_file_path_wav.exists():
            audio_path = audio_file_path_wav
        elif audio_file_path_flac.exists():
            audio_path = audio_file_path_flac
        else:
            # print(f"Could not find audio for {textgrid_file}")
            continue

        try:
            intervals = parse_textgrid(textgrid_file)
            y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            # print(f"Error loading or parsing {audio_path}: {e}")
            continue

        for interval in intervals:
            start_sample = int(interval['xmin'] * TARGET_SR)
            end_sample = int(interval['xmax'] * TARGET_SR)

            # ensure segment is not empty
            if start_sample >= end_sample:
                continue

            segment = y[start_sample:end_sample]

            # process and save segments longer than 0.5s
            if len(segment) > TARGET_SR * 0.5:
                processed_segment = process_audio_file(segment, TARGET_SR, TARGET_SAMPLES)

                if interval['label'] == '1':
                    filename = f'speech_{speech_count}.wav'
                    sf.write(speech_dir / filename, processed_segment, TARGET_SR)
                    all_files.append({'file': f'speech/{filename}', 'label': 1})
                    speech_count += 1
                else:
                    filename = f'no_speech_{no_speech_count}.wav'
                    sf.write(no_speech_dir / filename, processed_segment, TARGET_SR)
                    all_files.append({'file': f'no_speech/{filename}', 'label': 0})
                    no_speech_count += 1

    # Process test noise
    noise_base_path = Path('data/raw_datasets/noise-data-set/noise_train')
    noise_files = []
    for f in noise_base_path.rglob('*.wav'):
        if 'airconditioner' in f.stem.lower() or 'restaurant' in f.stem.lower():
            noise_files.append(f)

    for i, noise_file in enumerate(tqdm(noise_files, desc="Processing noise files")):
        y, sr = librosa.load(noise_file, sr=TARGET_SR, mono=True)

        # Split into 4 second chunks
        for j in range(0, len(y) - TARGET_SAMPLES, TARGET_SAMPLES):
            segment = y[j:j+TARGET_SAMPLES]
            processed_segment = process_audio_file(segment, TARGET_SR, TARGET_SAMPLES)
            filename = f'noise_{i}_{j}.wav'
            sf.write(test_noise_dir / filename, processed_segment, TARGET_SR)

    # Process general training noise (all noise from noise_train except airconditioner and restaurant)
    training_noise_count = 0
    general_noise_files = []
    for f in Path('data/raw_datasets/noise-data-set/noise_train').rglob('*.wav'):
        if 'airconditioner' not in f.stem and 'restaurant' not in f.stem:
            general_noise_files.append(f)

    for i, noise_file in enumerate(tqdm(general_noise_files, desc="Processing general training noise")):
        y, sr = librosa.load(noise_file, sr=TARGET_SR, mono=True)
        for j in range(0, len(y) - TARGET_SAMPLES, TARGET_SAMPLES // 2): # Overlapping chunks for more data
            segment = y[j:j+TARGET_SAMPLES]
            processed_segment = process_audio_file(segment, TARGET_SR, TARGET_SAMPLES)
            filename = f'general_noise_{i}_{j}.wav'
            sf.write(training_noise_dir / filename, processed_segment, TARGET_SR)
            training_noise_count += 1
    print(f"Created {training_noise_count} general training noise samples.")


    # Create metadata files
    df = pd.DataFrame(all_files)

    # Stratified split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

    train_df.to_csv(OUTPUT_BASE_DIR / 'train.csv', index=False)
    val_df.to_csv(OUTPUT_BASE_DIR / 'val.csv', index=False)
    test_df.to_csv(OUTPUT_BASE_DIR / 'test.csv', index=False)

    print(f"Created {len(train_df)} training samples, {len(val_df)} validation samples, and {len(test_df)} test samples.")


if __name__ == '__main__':
    organize_speech_data()
