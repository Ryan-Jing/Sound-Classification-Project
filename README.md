# Audio Classification with Dynamic Noise Curriculum Learning

A comprehensive audio classification system that combines multiple datasets and uses progressive noise curriculum learning to train robust models. Supports CNN, RNN/LSTM, and SVM models for comparison.

---

## 🎯 Features

- **Multi-Dataset Integration**: Combines UrbanSound8K, Speech Activity Detection, and multiple noise datasets
- **Dynamic Noise Mixing**: Mixes noise on-the-fly during training (no pre-mixed files stored)
- **Curriculum Learning**: Progressive noise training from clean audio to challenging noise levels
- **Multiple Models**: Compare CNN, RNN/LSTM, and SVM performance
- **Cross-Validation**: Uses UrbanSound8K's 10-fold structure to prevent data leakage
- **Automatic Validation**: Verifies all processed audio meets specifications

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# For .webm support (audio-noise-dataset):
# macOS: brew install ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
```

**Required packages:** PyTorch, librosa, soundfile, numpy, pandas, scikit-learn, tqdm

### 2. Download Datasets

Download these Kaggle datasets and place in `datasets/` folder:

- [UrbanSound8K](https://www.kaggle.com/datasets/chrisfilo/urbansound8k)
- [Speech Activity Detection](https://www.kaggle.com/datasets/lazyrac00n/speech-activity-detection-datasets/data)
- [Audio Noise Dataset](https://www.kaggle.com/datasets/minsithu/audio-noise-dataset)
- [Noise Dataset 2](https://www.kaggle.com/datasets/abdelrhamanfakhry/noise-data-set)

### 3. Organize Datasets

```bash
python audio_dataset_organizer.py
```

**Time:** 15-30 minutes
**Output:** `organized_dataset/` with ~8,780 clean samples + ~24,384 noise samples

### 4. Train Models

```bash
python training_pipeline.py
```

**Time:** 2-3 hours (GPU) or 8-12 hours (CPU) for deep learning models

---

## 📁 Project Structure

```
Sound-Classification-Project/
├── audio_dataset_organizer.py   # Dataset preprocessing (Step 1)
├── dynamic_noise_mixer.py       # Dynamic noise mixing & data loading
├── training_pipeline.py         # Model training (Step 2)
├── visualize_results.py         # Result visualization
├── quick_start.py               # Interactive setup script
├── requirements.txt             # Python dependencies
└── organized_dataset/           # Created after running organizer
    ├── clean_audio/
    │   ├── fold1/ ... fold10/   # ~8,780 .wav files (4s, 22050Hz)
    ├── noise_audio/             # ~24,384 .wav files
    └── metadata/
        ├── clean_samples.csv    # Sample metadata with labels
        ├── noise_samples.csv    # Noise file list
        └── class_mapping.csv    # Class ID → name mapping
```

---

## 🎓 Understanding the System

### Dataset Composition

**Clean Audio (Features to Classify):**
- **UrbanSound8K**: 8,732 samples across 10 classes
  - air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
- **Human Speech**: ~30-50 clean samples (NEW class #11)
  - From PTDB-TUG corpus (Female/Male folders only)
  - Pre-noisy Noizeus samples automatically excluded ✅

**Noise Audio (Background Corruption):**
- ~24,384 noise samples from 2 datasets
- Randomly selected and dynamically mixed during training

### How Dynamic Noise Mixing Works

**Traditional Approach (NOT used):**
```
10,000 samples × 6 noise levels × 4s × 22050Hz × 2 bytes = ~5.3 GB storage
```

**Our Approach:**
1. Load clean audio sample
2. Load random noise sample
3. Mix on-the-fly at current curriculum SNR level
4. Extract MFCC features
5. Feed to model

**Benefits:**
- Storage: Only ~2 GB (clean + noise)
- Infinite noise variations (different random noise each epoch)
- Easy to adjust noise levels without regenerating files

**SNR Mixing Formula:**
```python
signal_power = mean(clean_audio²)
noise_power_target = signal_power / (10^(SNR_dB/10))
noise_scaling = sqrt(noise_power_target / noise_power)
mixed_audio = clean_audio + (noise × noise_scaling)
```

### Curriculum Learning Schedule

| Epochs | SNR (dB) | Noise Level | Learning Rate |
|--------|----------|-------------|---------------|
| 0-4    | ∞ (clean) | None       | 0.001         |
| 5-9    | 25        | Very light | 0.0008        |
| 10-14  | 20        | Light      | 0.00064       |
| 15-19  | 15        | Moderate   | 0.000512      |
| 20-24  | 12        | Heavy      | 0.0004096     |
| 25-29  | 10        | Very heavy | 0.00032768    |

**Why this works:**
- Model learns basic patterns on clean data first
- Gradually becomes noise-robust
- Learning rate automatically decreases as task difficulty increases

---

## 🏗️ Model Architectures

### CNN (Convolutional Neural Network)
```
Input: MFCC features [batch, 40, 173]
↓
Conv2D(1→32) + BatchNorm + ReLU + MaxPool
Conv2D(32→64) + BatchNorm + ReLU + MaxPool
Conv2D(64→128) + BatchNorm + ReLU + MaxPool
↓
Flatten
FC(→256) + Dropout(0.3)
FC(256→128) + Dropout(0.3)
FC(128→11 classes)
```

### RNN/LSTM (Recurrent Neural Network)
```
Input: MFCC features [batch, 40, 173]
↓
Bidirectional LSTM (40→128, 2 layers)
Dropout(0.3)
FC(256→11 classes)
```

### SVM (Support Vector Machine)
- Input: Flattened MFCC features
- Kernel: RBF (Radial Basis Function)
- Feature scaling: StandardScaler
- Retrained at each curriculum stage

---

## 🎯 Expected Performance

| Model | Clean Audio | SNR 20dB | SNR 10dB | Training Time* | Notes |
|-------|-------------|----------|----------|----------------|-------|
| **CNN** | 85-90% | 80-85% | 70-75% | ~2 hours | Best overall |
| **RNN/LSTM** | 83-88% | 78-83% | 68-73% | ~3 hours | Good temporal modeling |
| **SVM** | 75-80% | 70-75% | 60-65% | ~30 min | Fast, lower accuracy |

*On GPU (NVIDIA RTX 3080 equivalent). CPU is ~4-5x slower.
**Random baseline: 9% (11 classes)**

---

## ⚙️ Customization

### Change Noise Curriculum

Edit `training_pipeline.py` lines 289-292:

```python
# Faster progression (more aggressive)
noise_curriculum = {
    'epochs': [0, 5, 10, 15],
    'snr_db': [float('inf'), 20, 15, 10]
}

# Gentler progression (slower)
noise_curriculum = {
    'epochs': [0, 10, 20, 30, 40],
    'snr_db': [float('inf'), 30, 25, 20, 15]
}

# No curriculum (constant noise at SNR 10dB)
noise_curriculum = {
    'epochs': [0],
    'snr_db': [10]
}

# No noise (baseline evaluation)
noise_curriculum = None
```

### Adjust Memory Usage

**If you get "CUDA out of memory" errors:**

```python
# training_pipeline.py line 445
batch_size=16  # Reduce from 32

# dynamic_noise_mixer.py line 340
n_mfcc=20  # Reduce from 40 (fewer MFCC coefficients)

# dynamic_noise_mixer.py line 67
if len(self.noise_cache) < 100:  # Reduce cache from 500 to 100
```

### Use Different Test Fold

```python
# Train on specific fold
results = train_with_curriculum(
    model_type='CNN',
    test_fold=2,  # Change from 1 to any fold 1-10
    num_epochs=30
)

# Full 10-fold cross-validation
all_results = []
for fold in range(1, 11):
    result = train_with_curriculum(model_type='CNN', test_fold=fold, num_epochs=30)
    all_results.append(result)
avg_accuracy = mean([r['epochs'][-1]['test_acc'] for r in all_results])
```

### Modify MFCC Parameters

Edit `dynamic_noise_mixer.py`:

```python
mfcc_transform = MFCCTransform(
    sr=22050,        # Sampling rate
    n_mfcc=40,       # Number of coefficients (try 20 or 80)
    n_fft=2048,      # FFT window size
    hop_length=512   # Smaller = more time frames
)
```

---

## 🐛 Troubleshooting

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| "Metadata file not found" | Wrong dataset path | Verify paths in `audio_dataset_organizer.py` lines 383-388 |
| "No noise files found" | Missing .webm support | Install ffmpeg: `brew install ffmpeg` |
| "CUDA out of memory" | Batch size too large | Reduce `batch_size` to 16 or 8 |
| "Training is too slow" | Using CPU | Use GPU (10x faster) or reduce dataset size |
| "Accuracy stuck at ~10%" | Model not learning | Check learning rate, data loading, class balance |

### Debug Tips

1. **Test with 1 epoch first:**
   ```python
   # training_pipeline.py line 445
   num_epochs=1  # Verify everything works
   ```

2. **Monitor first epoch:**
   - Batch shapes should match expected dimensions
   - Loss should decrease from ~2.3 to ~1.5
   - No NaN values in gradients

3. **Print debug info:**
   ```python
   print(f"Batch features shape: {features.shape}")
   print(f"Current SNR: {train_dataset.get_noise_level()}")
   ```

---

## 📊 Output Files

After training, these files are generated:

- `best_cnn_fold1.pth` - Best CNN model weights
- `best_rnn_fold1.pth` - Best RNN model weights
- `svm_fold1_stage*.pkl` - SVM models at each curriculum stage
- `cnn_results_fold1.json` - CNN training metrics
- `rnn_results_fold1.json` - RNN training metrics
- `svm_results_fold1.json` - SVM training results
- `model_comparison_fold1.json` - Performance comparison

### Visualize Results

```bash
python visualize_results.py
```

Generates:
- Training curves (accuracy/loss vs epoch)
- Curriculum impact visualization
- Model comparison charts
- Performance reports

---

## 🔧 Implementation Details

### What Was Fixed

All critical issues have been resolved:

✅ **Fixed UrbanSound8K paths** - Metadata and audio folder paths corrected
✅ **Added .webm support** - Now processes all noise files
✅ **Filter pre-noisy speech** - Excludes Noizeus corpus (already at SNR 5dB)
✅ **Dataset validation** - Automatically checks sample rate, duration, normalization
✅ **Optimized caching** - Increased to 500 noise samples (~44MB) for faster training

### What Was Already Correct

Your implementation had these aspects **perfectly designed**:

✅ Audio preprocessing (resampling, padding/cropping, normalization)
✅ Dynamic noise mixing (mathematically correct SNR calculations)
✅ Curriculum learning (progressive difficulty with LR adaptation)
✅ Fold structure (prevents data leakage)
✅ Model architectures (CNN, RNN/LSTM, SVM)
✅ MFCC extraction (proper configuration)
✅ Training pipeline (validation, checkpointing)

---

## 📈 Performance Tips

1. **Use GPU**: Training is ~10x faster on GPU vs CPU
2. **Cache optimization**: First 500 noise samples cached automatically (~44MB RAM)
3. **Parallel data loading**: Adjust `num_workers=4` in `dynamic_noise_mixer.py` line 401
4. **Start small**: Test with 1 epoch to verify setup before full training
5. **Compare baseline**: Train without noise first to establish baseline performance

---

## 💡 Advanced Usage

### Add TensorBoard Logging

```python
# Add to training_pipeline.py
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# In training loop:
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Accuracy/test', test_acc, epoch)
```

### Mixed Precision Training (faster on modern GPUs)

```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(features)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Include Pre-Noisy Speech (Optional)

```python
# In audio_dataset_organizer.py main()
organizer.organize_speech_dataset(
    SPEECH_DATASET_PATH,
    fold_assignment='balanced',
    exclude_noisy=False  # Include Noizeus samples
)
# Adds ~660 more speech samples (already at SNR 5dB)
```

---

## 📚 File Reference

| File | Purpose | When to Edit |
|------|---------|--------------|
| `audio_dataset_organizer.py` | Preprocess and organize datasets | Change paths, target SR/duration |
| `dynamic_noise_mixer.py` | Dynamic noise mixing & data loading | Adjust MFCC params, cache size |
| `training_pipeline.py` | Model training with curriculum | Change curriculum, hyperparameters, architecture |
| `visualize_results.py` | Result visualization | After training for analysis |
| `quick_start.py` | Interactive setup wizard | First-time project setup |
| `requirements.txt` | Python dependencies | Add new packages |

---

## 🎓 Citation

If you use this code for research, please cite the datasets:

**UrbanSound8K:**
```
J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research",
22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.
```

---

## 📝 License

This project is for educational and research purposes.

---

## ✨ Production Ready!

This implementation is ready for:

1. ✅ Academic research and experimentation
2. ✅ Model comparison studies (CNN vs RNN vs SVM)
3. ✅ Noise robustness evaluation
4. ✅ Curriculum learning research
5. ✅ Audio classification benchmarking

**All systems operational. Happy training! 🚀**
