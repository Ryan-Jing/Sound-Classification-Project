import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import warnings
import pandas as pd

# Import model classes
from models.cnn import CNNModel
from models.rnn import RNNModel

def process_audio(path, target_sr=16000, target_duration=3.0):
    """
    Loads, resamples, and normalizes a single audio file for inference.
    Pads or crops to the target duration.
    """
    target_samples = int(target_sr * target_duration)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, sr = librosa.load(path, sr=target_sr, mono=True)

        # Pad or crop
        if len(waveform) < target_samples:
            # Pad with silence
            pad_length = target_samples - len(waveform)
            waveform = np.pad(waveform, (0, pad_length), mode='constant')
        elif len(waveform) > target_samples:
            # Crop from the center
            start = (len(waveform) - target_samples) // 2
            waveform = waveform[start:start + target_samples]

        # Convert to a PyTorch tensor and add channel dimension
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        return waveform

    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

def get_spectrogram(waveform, target_sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    """
    Converts a waveform tensor to a Mel spectrogram tensor.
    """
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        n_mels=n_mels,
        hop_length=hop_length
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    mel_spec = mel_spectrogram_transform(waveform)
    mel_spec_db = amplitude_to_db(mel_spec)
    return mel_spec_db

def main():
    # --- MANUAL CONFIGURATION ---
    # 1. Set the model type you used for training ('CNN' or 'RNN')
    model_type = 'CNN'

    # 2. Set the path to your trained model weights file
    model_weights_path = Path('results/CNN_human_speech_20251215_101630/weights/CNN_human_speech_20251215_101630_final.pth')

    # 3. Set the path to the audio clip you want to classify
    audio_clip_path = Path('no_voice_test.wav')

    # 4. Set the path to the class mapping file from the same results folder
    class_mapping_path = Path('results/CNN_human_speech_20251215_101630/class_mapping.csv')
    # --------------------------

    # --- 1. Load class mapping ---
    if not class_mapping_path.is_file():
        print(f"Error: Class mapping file not found at {class_mapping_path}")
        return

    class_df = pd.read_csv(class_mapping_path)
    idx_to_class = pd.Series(class_df.class_name.values, index=class_df.class_id).to_dict()
    num_classes = len(idx_to_class)

    # --- 2. Initialize Model ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_type == 'CNN':
        model = CNNModel(num_classes=num_classes)
    elif model_type == 'RNN':
        model = RNNModel(input_size=128, num_classes=num_classes)
    else:
        print(f"Error: Model type '{model_type}' not supported for inference.")
        return

    # --- 3. Load Weights ---
    if not model_weights_path.is_file():
        print(f"Error: Model weights file not found at {model_weights_path}")
        return

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 4. Process Audio ---
    if not audio_clip_path.is_file():
        print(f"Error: Audio clip not found at {audio_clip_path}")
        return

    waveform = process_audio(audio_clip_path)
    if waveform is None:
        return

    # --- 5. Get Spectrogram ---
    spectrogram = get_spectrogram(waveform)
    spectrogram = spectrogram.unsqueeze(0).to(device) # Add batch dimension

    # --- 6. Run Inference ---
    with torch.no_grad():
        outputs = model(spectrogram)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)

    predicted_class = idx_to_class.get(predicted_idx.item(), "Unknown class")
    confidence = probabilities[0][predicted_idx.item()].item()

    print(f"\nPrediction: '{predicted_class}' with {confidence:.2%} confidence.")

if __name__ == '__main__':
    main()