# Sound Classification Pipeline

## Overview

This project provides a comprehensive pipeline for training, evaluating, and running inference with sound classification models. It is designed to be robust against noise by using a curriculum learning strategy, where models are progressively trained on data with increasing levels of background noise. The entire process is interactive and modular, allowing for easy experimentation with different audio features and model architectures (CNN, RNN, SVM).

## Features

- **Automated Data Processing**: A script (`new_audio_dataset_organizer.py`) processes raw audio from multiple sources into a structured, augmented dataset with varying noise levels.
- **Interactive Training**: The main `training_pipeline.py` script prompts the user to select which sound classes to train on and which model to use.
- **Multi-Model Support**: Includes implementations for a Convolutional Neural Network (CNN), a Recurrent Neural Network (RNN), and a baseline Support Vector Machine (SVM).
- **Curriculum Learning**: Improves noise robustness by training models first on clean audio, then incrementally on noisier samples while adjusting the learning rate.
- **Rigorous Evaluation**: Automatically runs 5 independent trials for each experiment to quantify model variability and performance.
- **Advanced Visualization**: Generates plots for learning curves with 95% confidence intervals and aggregated confusion matrices over all trials. It also saves example spectrograms to help visualize the model's input.
- **Inference Ready**: Saves the final model weights and provides an `inference.py` script to easily classify new, single audio files.

## Project Structure

```
/
├── organized_audio_datasets/   # -> Processed and augmented data appears here.
├── results/                    # -> All outputs (plots, weights, logs) are saved here.
├── models/                     # -> Contains model architectures (cnn.py, rnn.py).
├── trainers/                   # -> Contains the training logic for different models.
├── utils/                      # -> Contains helper scripts for visualization.
├── data/                       # -> Contains all the raw, unprocessed datasets.
|
├── new_audio_dataset_organizer.py # -> Script to prepare the dataset from raw sources.
├── training_pipeline.py        # -> Main script to run the interactive training process.
├── inference.py                # -> Script to classify a single audio file with a trained model.
├── requirements.txt            # -> Project dependencies.
└── README.md                   # -> This file.
```

## Workflow

Follow these steps to set up the environment and run the full pipeline.

### 1. Setup

First, install all the necessary Python packages using the `requirements.txt` file. It is highly recommended to do this within a virtual environment.

```bash
pip install -r requirements.txt
```

### 2. Step 1: Organize Dataset

Before training, you must process the raw audio files into a structured dataset. This script handles resampling, duration normalization (3 seconds), and the creation of noisy data augmentations.

Run the following command from the project root. This only needs to be done once.

```bash
python3 new_audio_dataset_organizer.py
```
This will create the `organized_audio_datasets/` directory containing all the processed files, sorted by class and noise level.

### 3. Step 2: Run Training & Evaluation

This is the main script to train and evaluate a model. It will run a complete experiment, including 5 independent trials, and save all results.

```bash
python3 training_pipeline.py
```

The script will interactively prompt you to:
1.  Select which sound class(es) to train on.
2.  Select the model architecture to use (CNN, RNN, or SVM).

All results, including plots, logs, model weights, and class mappings, will be saved in a new, timestamped folder inside the `results/` directory.

### 4. Step 3: Run Inference

After training a model, you can use the `inference.py` script to classify a single audio file.

1.  Open the `inference.py` script in a text editor.
2.  In the "MANUAL CONFIGURATION" section, set the paths to point to your desired:
    - `model_weights_path` (the `.pth` file inside `results/.../weights/`)
    - `audio_clip_path` (the `.wav` file you want to test)
    - `class_mapping_csv` (the `.csv` file in your results folder)
3.  Run the script:

```bash
python3 inference.py
```

The script will print the predicted class and the model's confidence.