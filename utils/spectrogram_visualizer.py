import matplotlib.pyplot as plt
import librosa
import numpy as np
from pathlib import Path
import torch

def save_spectrogram(spec_tensor, save_path, title):
    """
    Saves a single spectrogram tensor to an image file.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    # Squeeze to remove channel and batch dimensions for plotting
    spec_numpy = spec_tensor.squeeze().cpu().numpy()
    
    # Use librosa.display.specshow to plot the spectrogram
    img = librosa.display.specshow(spec_numpy, ax=ax, y_axis='mel', x_axis='time', sr=16000)
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_example_spectrograms(dataloader, class_names, output_dir, num_examples=5):
    """
    Pulls a few examples from the dataloader and saves their spectrograms.
    """
    example_dir = output_dir / 'example_spectrograms'
    example_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving {num_examples} example spectrograms to {example_dir}...")

    # Get a few samples
    examples_found = 0
    for specs, labels in dataloader:
        for i in range(len(specs)):
            if examples_found >= num_examples:
                break
            
            spec = specs[i]
            label_idx = labels[i].item()
            class_name = class_names[label_idx]
            
            save_path = example_dir / f"example_{class_name}_{examples_found}.png"
            title = f"Example Spectrogram for Class: {class_name}"
            save_spectrogram(spec, save_path, title)
            examples_found += 1
        
        if examples_found >= num_examples:
            break
