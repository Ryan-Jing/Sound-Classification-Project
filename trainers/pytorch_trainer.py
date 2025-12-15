import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from pathlib import Path

from data_loader import prepare_dataloaders

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc="  Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = float(correct_predictions) / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="  Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = float(correct_predictions) / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc

def train_pytorch_model(model, dataloaders, learning_rates, output_dir, num_epochs_per_level, model_name, class_names):
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    noise_levels = ['clean', '20dB', '15dB', '10dB', '5dB']

    for level in noise_levels:
        print(f"\n--- Curriculum Level: {level} ---")

        # Set learning rate for this level
        lr = learning_rates[level]
        optimizer = Adam(model.parameters(), lr=lr)
        print(f"Set learning rate to: {lr}")

        # Get the specific data for this level
        # Note: This is a simplified approach for demonstration. A more robust implementation
        # might create separate DataLoaders for each noise level. Here we use the same split.
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']

        if not train_loader or not val_loader:
            print(f"No data for level {level}. Skipping.")
            continue

        epoch_iterator = tqdm(range(num_epochs_per_level), desc=f"Level: {level} (Epochs)")
        for epoch in epoch_iterator:
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            epoch_iterator.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{train_acc:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Val Acc': f'{val_acc:.4f}'
            })

    print("\n--- Testing Final Model ---")
    test_loader = dataloaders['test']
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # --- Save Model Weights ---
    print("\n--- Saving Model Weights ---")
    weights_dir = output_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    save_path = weights_dir / f"{model_name}_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

    return history, np.array(all_labels), np.array(all_preds)
