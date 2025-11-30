"""
Complete Training Pipeline for Audio Classification
Supports CNN, SVM, and RNN/LSTM models with dynamic noise curriculum learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import json
from pathlib import Path
from tqdm import tqdm
import pickle

from dynamic_noise_mixer import create_dataloaders, AudioDatasetWithDynamicNoise, MFCCTransform


# ============================================================================
# Model Definitions
# ============================================================================

class AudioCNN(nn.Module):
    """CNN model for audio classification from MFCC features."""
    
    def __init__(self, num_classes=10, n_mfcc=40):
        super(AudioCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Will be set dynamically based on input size
        self.fc1 = None
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: [batch, n_mfcc, time_frames]
        # Add channel dimension
        x = x.unsqueeze(1)  # [batch, 1, n_mfcc, time_frames]
        
        # Convolutional blocks
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Initialize fc1 if needed
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)
        
        # Fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x


class AudioRNN(nn.Module):
    """RNN/LSTM model for audio classification from MFCC features."""
    
    def __init__(self, num_classes=10, n_mfcc=40, hidden_size=128, 
                 num_layers=2, rnn_type='LSTM', bidirectional=True):
        super(AudioRNN, self).__init__()
        
        self.n_mfcc = n_mfcc
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # RNN layer (LSTM or basic RNN)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=n_mfcc,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.3 if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.RNN(
                input_size=n_mfcc,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0.3 if num_layers > 1 else 0
            )
        
        # Output layer
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: [batch, n_mfcc, time_frames]
        # Transpose for RNN: [batch, time_frames, n_mfcc]
        x = x.transpose(1, 2)
        
        # RNN forward pass
        # output shape: [batch, time_frames, hidden_size * num_directions]
        output, _ = self.rnn(x)
        
        # Use the last time step output
        output = output[:, -1, :]
        
        # Fully connected layer
        output = self.dropout(output)
        output = self.fc(output)
        
        return output


# ============================================================================
# Training Functions
# ============================================================================

class CNNTrainer:
    """Trainer for CNN model with curriculum learning."""
    
    def __init__(self, model, device, num_classes=10):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader, optimizer, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for features, labels in pbar:
            features, labels = features.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        return total_loss / len(train_loader), 100. * correct / total
    
    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        return accuracy, all_preds, all_labels


class SVMTrainer:
    """Trainer for SVM model with curriculum learning."""
    
    def __init__(self, C=1.0, kernel='rbf', gamma='scale'):
        self.model = SVC(C=C, kernel=kernel, gamma=gamma, verbose=False)
        self.scaler = StandardScaler()
        
    def extract_features(self, dataloader):
        """Extract flattened MFCC features and labels."""
        features_list = []
        labels_list = []
        
        for features, labels in tqdm(dataloader, desc='Extracting features'):
            # Flatten MFCC features: [batch, n_mfcc, time] -> [batch, n_mfcc * time]
            batch_size = features.size(0)
            features_flat = features.view(batch_size, -1).numpy()
            
            features_list.append(features_flat)
            labels_list.append(labels.numpy())
        
        X = np.vstack(features_list)
        y = np.concatenate(labels_list)
        
        return X, y
    
    def train(self, train_loader):
        """Train SVM model."""
        print("Extracting training features...")
        X_train, y_train = self.extract_features(train_loader)
        
        print("Scaling features...")
        X_train = self.scaler.fit_transform(X_train)
        
        print(f"Training SVM on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        
        # Training accuracy
        train_preds = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds) * 100
        
        return train_acc
    
    def evaluate(self, test_loader):
        """Evaluate SVM model."""
        print("Extracting test features...")
        X_test, y_test = self.extract_features(test_loader)
        
        print("Scaling features...")
        X_test = self.scaler.transform(X_test)
        
        print("Evaluating...")
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions) * 100
        
        return accuracy, predictions, y_test


# ============================================================================
# Main Training Pipeline
# ============================================================================

def train_with_curriculum(model_type='CNN', organized_dataset_dir='organized_dataset',
                         test_fold=1, num_epochs=30, batch_size=32, 
                         learning_rate=0.001, device='cuda'):
    """
    Complete training pipeline with progressive noise curriculum.
    
    Args:
        model_type: 'CNN', 'SVM', or 'RNN'
        organized_dataset_dir: Path to organized dataset
        test_fold: Which fold to use for testing
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate (for CNN/RNN)
        device: 'cuda' or 'cpu'
    """
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load class mapping to get number of classes
    import pandas as pd
    class_mapping_path = Path(organized_dataset_dir) / 'metadata' / 'class_mapping.csv'
    class_df = pd.read_csv(class_mapping_path)
    num_classes = len(class_df)
    print(f"Number of classes: {num_classes}")
    
    # Define noise curriculum
    # Progressive noise: clean -> SNR 25dB -> 20dB -> 15dB -> 12dB -> 10dB
    noise_curriculum = {
        'epochs': [0, 5, 10, 15, 20, 25],
        'snr_db': [float('inf'), 25, 20, 15, 12, 10]
    }
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, test_loader, train_dataset = create_dataloaders(
        organized_dataset_dir=organized_dataset_dir,
        test_fold=test_fold,
        batch_size=batch_size,
        noise_curriculum=noise_curriculum,
        num_workers=4
    )
    
    # Initialize model and trainer based on type
    if model_type == 'CNN':
        print("\nInitializing CNN model...")
        model = AudioCNN(num_classes=num_classes, n_mfcc=40)
        trainer = CNNTrainer(model, device, num_classes)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
    elif model_type == 'RNN':
        print("\nInitializing RNN/LSTM model...")
        model = AudioRNN(num_classes=num_classes, n_mfcc=40, 
                        hidden_size=128, num_layers=2, rnn_type='LSTM')
        trainer = CNNTrainer(model, device, num_classes)  # Same training logic
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
    elif model_type == 'SVM':
        print("\nInitializing SVM model...")
        trainer = SVMTrainer(C=1.0, kernel='rbf', gamma='scale')
        # Note: SVM will be retrained each curriculum stage
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Training loop with curriculum
    results = {
        'model_type': model_type,
        'test_fold': test_fold,
        'epochs': [],
        'curriculum_stages': []
    }
    
    print("\n" + "=" * 70)
    print(f"Starting {model_type} training with noise curriculum")
    print("=" * 70)
    
    if model_type in ['CNN', 'RNN']:
        # PyTorch model training with epochs
        best_test_acc = 0
        
        for epoch in range(num_epochs):
            # Update curriculum
            train_dataset.set_epoch(epoch)
            snr = train_dataset.get_noise_level()
            
            if snr is None:
                print(f"\nEpoch {epoch+1}/{num_epochs} - Training with CLEAN audio")
            else:
                print(f"\nEpoch {epoch+1}/{num_epochs} - Training with SNR = {snr} dB")
                # Reduce learning rate when noise increases
                if epoch in noise_curriculum['epochs'][1:]:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.8
                        print(f"  Reduced learning rate to {param_group['lr']:.6f}")
            
            # Train
            train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, epoch)
            
            # Evaluate
            test_acc, test_preds, test_labels = trainer.evaluate(test_loader)
            
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Acc: {test_acc:.2f}%")
            
            # Save results
            results['epochs'].append({
                'epoch': epoch,
                'snr_db': snr,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc
            })
            
            # Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), f'best_{model_type.lower()}_fold{test_fold}.pth')
                print(f"  New best model saved! (Test Acc: {best_test_acc:.2f}%)")
            
            # Step scheduler
            scheduler.step()
    
    else:  # SVM
        # Train SVM at each curriculum stage
        for stage_idx, (epoch_start, snr) in enumerate(zip(
            noise_curriculum['epochs'], noise_curriculum['snr_db']
        )):
            train_dataset.set_epoch(epoch_start)
            
            if snr == float('inf'):
                print(f"\nStage {stage_idx+1} - Training with CLEAN audio")
            else:
                print(f"\nStage {stage_idx+1} - Training with SNR = {snr} dB")
            
            # Train SVM
            train_acc = trainer.train(train_loader)
            
            # Evaluate
            test_acc, test_preds, test_labels = trainer.evaluate(test_loader)
            
            print(f"  Train Acc: {train_acc:.2f}%")
            print(f"  Test Acc: {test_acc:.2f}%")
            
            # Save results
            results['curriculum_stages'].append({
                'stage': stage_idx,
                'snr_db': snr,
                'train_acc': train_acc,
                'test_acc': test_acc
            })
            
            # Save model at this stage
            with open(f'svm_fold{test_fold}_stage{stage_idx}.pkl', 'wb') as f:
                pickle.dump({'model': trainer.model, 'scaler': trainer.scaler}, f)
    
    # Save final results
    results_path = f'{model_type.lower()}_results_fold{test_fold}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Results saved to {results_path}")
    print("=" * 70)
    
    return results


def compare_all_models(organized_dataset_dir='organized_dataset', test_fold=1):
    """Train and compare CNN, SVM, and RNN models."""
    
    results = {}
    
    # Train CNN
    print("\n" + "=" * 70)
    print("TRAINING CNN MODEL")
    print("=" * 70)
    results['CNN'] = train_with_curriculum(
        model_type='CNN',
        organized_dataset_dir=organized_dataset_dir,
        test_fold=test_fold,
        num_epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Train RNN
    print("\n" + "=" * 70)
    print("TRAINING RNN/LSTM MODEL")
    print("=" * 70)
    results['RNN'] = train_with_curriculum(
        model_type='RNN',
        organized_dataset_dir=organized_dataset_dir,
        test_fold=test_fold,
        num_epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Train SVM
    print("\n" + "=" * 70)
    print("TRAINING SVM MODEL")
    print("=" * 70)
    results['SVM'] = train_with_curriculum(
        model_type='SVM',
        organized_dataset_dir=organized_dataset_dir,
        test_fold=test_fold,
        batch_size=32
    )
    
    # Save comparison
    with open(f'model_comparison_fold{test_fold}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED - COMPARISON COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    # Example: Train a single model
    # results = train_with_curriculum(
    #     model_type='CNN',  # or 'RNN' or 'SVM'
    #     organized_dataset_dir='organized_dataset',
    #     test_fold=1,
    #     num_epochs=30
    # )
    
    # Example: Compare all models
    results = compare_all_models(
        organized_dataset_dir='organized_dataset',
        test_fold=1
    )
