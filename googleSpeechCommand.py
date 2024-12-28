import os
import pandas as pd
import shutil
import time
import numpy as np
import torch
import librosa
import torchaudio.transforms as T
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import hf_hub_download
from speechbrain.inference import EncoderClassifier
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, classification_report


# SYMLINK uyarısını devre dışı bırak
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"

# Hugging Face dosyalarını indir ve elle kopyala
def download_and_copy_model(repo_id, local_dir):
    files = [
        "classifier.ckpt",
        "embedding_model.ckpt",
        "mean_var_norm_emb.ckpt",
        "hyperparams.yaml",
    ]
    os.makedirs(local_dir, exist_ok=True)

    for file in files:
        file_path = hf_hub_download(repo_id=repo_id, filename=file)
        shutil.copy(file_path, os.path.join(local_dir, file))

# Model dosyalarını indir ve tmp_model'e kopyala
download_and_copy_model("speechbrain/spkrec-xvect-voxceleb", "tmp_model")

# SpeechBrain modelini yükle
classifier = EncoderClassifier.from_hparams(source="tmp_model")

# Cihaz seçimi
device = "cuda" if torch.cuda.is_available() else "cpu"
classifier.to(device)

def preprocess_waveform(waveform):
    # MelSpectrogram dönüştürücü
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=24
    ).to(device)
    mel_spec = mel_spectrogram(waveform)
    mel_spec = mel_spec.permute(0, 2, 1)  # [batch, freq, time] -> [batch, time, freq]
    return mel_spec

# GTZAN dataset class
class GTZANDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# GTZAN veri setini yükle
def load_gtzan_dataset(audio_path):
    filepaths = []
    labels = []
    for genre in os.listdir(audio_path):
        genre_path = os.path.join(audio_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                filepaths.append(os.path.join(genre_path, file))
                labels.append(genre)

    label_map = {genre: idx for idx, genre in enumerate(sorted(set(labels)))}
    numeric_labels = [label_map[label] for label in labels]

    data = []
    valid_labels = []

    for file_path, label in zip(filepaths, numeric_labels):
        try:
            waveform, sr = librosa.load(file_path, sr=16000)
            if len(waveform) < 16000:
                waveform = np.pad(waveform, (0, 16000 - len(waveform)), mode='constant')
            else:
                waveform = waveform[:16000]
            data.append(waveform)
            valid_labels.append(label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return np.array(data), np.array(valid_labels)

def train_speech_brain():
    audio_path = "archive/Data/genres_original"  # Ses dosyalarının bulunduğu dizin

    # Veri setini yükle
    data, labels = load_gtzan_dataset(audio_path)

    # Embedding çıkış boyutunu öğren
    embedding_output_size = classifier.mods["embedding_model"].blocks[-1].w.out_features
    print(f"Embedding Output Size: {embedding_output_size}")

    # Sınıflandırıcıyı yeniden tanımla
    num_classes = len(set(labels))
    classifier.mods["classifier"] = torch.nn.Linear(embedding_output_size, num_classes).to(device)

    # K-Fold Cross-Validation
    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    total_time = 0
    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(data, labels)):
        print(f"Fold {fold + 1}")

        # Split into train-val-test
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=42)

        train_data, train_labels = data[train_idx], labels[train_idx]
        val_data, val_labels = data[val_idx], labels[val_idx]
        test_data, test_labels = data[test_idx], labels[test_idx]

        print(f"Fold {fold + 1}: Train Size: {len(train_data)}, Validation Size: {len(val_data)}, Test Size: {len(test_data)}")

        # Create DataLoaders
        train_loader = DataLoader(GTZANDataset(train_data, train_labels), batch_size=32, shuffle=True)
        val_loader = DataLoader(GTZANDataset(val_data, val_labels), batch_size=32, shuffle=False)
        test_loader = DataLoader(GTZANDataset(test_data, test_labels), batch_size=32, shuffle=False)

        # Model optimizer ve kayıp fonksiyonu
        optimizer = torch.optim.Adam(classifier.mods["embedding_model"].parameters(), lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        start_time = time.time()
        # Eğitim döngüsü
        for epoch in range(50):
            classifier.train()
            train_loss = 0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1} Training") as pbar:
                for batch in pbar:
                    optimizer.zero_grad()

                    inputs, targets = batch
                    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
                    targets = torch.tensor(targets, dtype=torch.long).to(device)

                    # Girdi işleme ve modelden geçirme
                    inputs = preprocess_waveform(inputs)
                    embeddings = classifier.mods["embedding_model"](inputs)
                    outputs = classifier.mods["classifier"](embeddings)
                    # Gereksiz boyutu kaldır
                    outputs = outputs.squeeze(1)
                    # Kayıp hesapla ve geri yayılım yap
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    pbar.set_postfix(loss=train_loss / (pbar.n + 1))

            # Doğrulama döngüsü
            classifier.eval()
            val_loss = 0
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation") as pbar:
                    for batch in pbar:
                        inputs, targets = batch
                        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
                        targets = torch.tensor(targets, dtype=torch.long).to(device)

                        inputs = preprocess_waveform(inputs)
                        embeddings = classifier.mods["embedding_model"](inputs)
                        outputs = classifier.mods["classifier"](embeddings)
                        outputs = outputs.squeeze(1)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        pbar.set_postfix(loss=val_loss / (pbar.n + 1))

            print(f" Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # En iyi modelin ağırlıklarını kaydet
                best_model_weights = classifier.state_dict()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                # En iyi modelin ağırlıklarını geri yükle
                classifier.load_state_dict(best_model_weights)
                break

        # Test döngüsü
        classifier.eval()
        test_targets = []
        test_predictions = []

        with torch.no_grad():
            with tqdm(test_loader, desc=f"Fold {fold + 1} Testing") as pbar:
                for batch in pbar:
                    inputs, targets = batch
                    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
                    targets = torch.tensor(targets, dtype=torch.long).to(device)

                    inputs = preprocess_waveform(inputs)
                    embeddings = classifier.mods["embedding_model"](inputs)
                    outputs = classifier.mods["classifier"](embeddings)
                    outputs = outputs.squeeze(1)
                    predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                    test_targets.extend(targets.cpu().numpy())
                    test_predictions.extend(predictions)

        duration = time.time() - start_time
        total_time += duration

        # Performans metrikleri
        test_accuracy = accuracy_score(test_targets, test_predictions)
        test_precision = precision_score(test_targets, test_predictions, average="weighted")
        test_recall = recall_score(test_targets, test_predictions, average="weighted")
        test_f1 = f1_score(test_targets, test_predictions, average="weighted")
        print(f"Fold {fold + 1}: test_accuracy: {test_accuracy:.4f}, test_precision: {test_precision:.4f},test_recall: {test_recall:.4f}, test_f1: {test_f1:.4f}")

        # # Confusion Matrix ve Classification Report
        # print("Confusion Matrix:")
        # print(confusion_matrix(test_targets, test_predictions))
        # print("Classification Report:")
        # print(classification_report(test_targets, test_predictions, zero_division=0))

        fold_metrics.append({
            "Fold": fold + 1,
            "Accuracy": test_accuracy,
            "Precision": test_precision,
            "Recall": test_recall,
            "F1 Score": test_f1,
            "Duration (s)": duration
        })

    fold_metrics = pd.DataFrame(fold_metrics)
    return fold_metrics, total_time

# fold_metrics, total_time = train_speech_brain()
# # Sonuçları yazdır
# print(fold_metrics)
# print(f"Toplam süre: {total_time:.2f} saniye")
