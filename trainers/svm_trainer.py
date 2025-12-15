import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

def extract_svm_features(file_path, target_sr=16000, n_mfcc=20):
    """
    Extracts aggregated MFCC features from a single audio file.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)
        
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
        
        # Aggregate features over time
        mean_mfccs = np.mean(mfccs, axis=1)
        std_mfccs = np.std(mfccs, axis=1)
        
        return np.concatenate([mean_mfccs, std_mfccs])
    except Exception:
        # Return None if a file is corrupt or can't be processed
        return None

def train_svm_model(dataloaders, class_names):
    """
    Trains an SVM model on the provided data.
    
    Args:
        dataloaders (dict): A dictionary containing 'train' and 'test' keys,
                            with values being tuples of (file_paths, labels).
        class_names (list): A list of the class names for reporting.

    Returns:
        A tuple of (test_labels, test_predictions).
    """
    train_files, train_labels = dataloaders['train']
    test_files, test_labels = dataloaders['test']
    
    # --- Extract training features ---
    print("Extracting features for training set...")
    X_train = []
    y_train = []
    for file, label in tqdm(zip(train_files, train_labels), total=len(train_files)):
        features = extract_svm_features(file)
        if features is not None:
            X_train.append(features)
            y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # --- Extract testing features ---
    print("Extracting features for test set...")
    X_test = []
    y_test = []
    for file, label in tqdm(zip(test_files, test_labels), total=len(test_files)):
        features = extract_svm_features(file)
        if features is not None:
            X_test.append(features)
            y_test.append(label)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # --- Feature Scaling ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train SVM Model ---
    print("Training SVM classifier...")
    # Using a radial basis function (RBF) kernel, which is a good default
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
    model.fit(X_train_scaled, y_train)

    # --- Evaluate Model ---
    print("Evaluating model on the test set...")
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Test Accuracy: {accuracy:.4f}")

    return y_test, y_pred
