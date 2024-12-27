import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
import pandas as pd
import librosa
from tqdm import tqdm
from torch.cuda.amp import autocast
import gc
import time  # Zaman ölçümü için gerekli modül

from torch.utils.data import DataLoader, TensorDataset
# OpenAI Whisper Model Ayarları
model_name = "openai/whisper-tiny"  # Daha küçük model kullanımı
processor = WhisperProcessor.from_pretrained(model_name)
# model = WhisperForConditionalGeneration.from_pretrained(model_name)

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)

# Ortam değişkeni ile bellek yönetimini optimize edin
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Audio dosyalarının bulunduğu dizin
audio_path = "archive/Data/genres_original"
batch_size = 1  # Batch boyutunu küçültüldü
gradient_accumulation_steps = 8 # Gradyan toplama adımları
num_folds = 5
early_stopping_patience = 3  # Early stopping için sabırlı bekleme süresi
if not os.path.exists("best_models"):
    os.makedirs("best_models")

def preprocess_audio(file_path, target_sr=16000):
    try:
        # Ses dosyasını yükle
        audio, sr = librosa.load(file_path, sr=target_sr)
        input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

        # Giriş verilerini float32'de bırak
        return input_features
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def filter_data(filepaths, labels):
    filtered_inputs = []
    filtered_labels = []
    for fp, label in zip(filepaths, labels):
        input_features = preprocess_audio(fp)
        if input_features is not None:
            filtered_inputs.append(input_features)
            # Her bir label'i tensor olarak dönüştürün
            filtered_labels.append(torch.tensor([label], dtype=torch.long))
    return filtered_inputs, filtered_labels

def prepare_labels(labels):
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return labels_tensor.unsqueeze(1)


def train_openaiwhisper():
    
    # Veri kümesi hazırlanıyor
    filepaths = []
    labels = []
    for label in os.listdir(audio_path):
        class_path = os.path.join(audio_path, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                filepaths.append(os.path.join(class_path, file))
                labels.append(label)

    data = pd.DataFrame({'filepath': filepaths, 'label': labels})
    label_map = {label: idx for idx, label in enumerate(data['label'].unique())}
    data['label_encoded'] = data['label'].map(label_map)

    # Stratified K-Fold ile veri ayrımı
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    total_duration=0
    metrics_summary = []  # Her fold'un metriklerini saklayacağımız liste
    for fold, (train_idx, test_idx) in enumerate(kf.split(data['filepath'], data['label_encoded'])):
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        # print(torch.cuda.memory_summary())
        print(f"\nModel {model_name} - Training on Fold {fold + 1}/{num_folds}")

        # Eğitim, doğrulama ve test ayrımı
        test_data = data.iloc[test_idx]
        temp_data = data.iloc[train_idx]
        train_data, val_data = train_test_split(temp_data, test_size=0.1, stratify=temp_data['label_encoded'], random_state=42)

        # Veri filtreleme
        train_inputs, train_labels = filter_data(train_data['filepath'], train_data['label_encoded'])
        val_inputs, val_labels = filter_data(val_data['filepath'], val_data['label_encoded'])
        test_inputs, test_labels = filter_data(test_data['filepath'], test_data['label_encoded'])

        # Tensor olarak birleştirilir
        train_inputs = torch.cat(train_inputs).to(device)
        train_labels = prepare_labels(train_labels).to(device)
        val_inputs = torch.cat(val_inputs).to(device)
        val_labels = prepare_labels(val_labels).to(device)
        test_inputs = torch.cat(test_inputs).to(device)
        test_labels = prepare_labels(test_labels).to(device)

        # print("Input min:", torch.min(train_inputs), "max:", torch.max(train_inputs))
        # print("Input mean:", torch.mean(train_inputs))
        # print("Labels unique:", torch.unique(train_labels))

        # Modeli eğit
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Early stopping için değişkenler
        best_val_loss = float('inf')
        patience_counter = 0

        # Eğitim süresini ölçmek için başlangıç zamanı
        start_time = time.time()

        for epoch in range(50):  # Max 50 epoch
            print(f"Epoch {epoch + 1}/50")
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0

            # Eğitim progress bar
            train_progress = tqdm(enumerate(zip(train_inputs.split(batch_size), train_labels.split(batch_size))), 
                                   total=len(train_inputs) // batch_size, 
                                   desc="Training", 
                                   leave=True)

            for step, (inputs, labels) in train_progress:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                # with autocast():  # float16 otomatik geçişi
                outputs = model(input_features=inputs, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps

                loss.backward()
                # print("Logits shape:", outputs.logits.shape)
                # print("Labels shape:", labels.shape)

                running_loss += loss.item()

                # Doğruluk hesaplama
                preds = torch.argmax(outputs.logits, dim=-1)
                train_correct += (preds == labels.squeeze()).sum().item()
                train_total += labels.size(0)

                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_inputs) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache() 

                train_progress.set_postfix({
                    "loss": running_loss / (step + 1),
                    "accuracy": train_correct / train_total
                })

            epoch_loss = running_loss / len(train_inputs)
            epoch_accuracy = train_correct / train_total
            print(f"Epoch {epoch + 1}: Training Loss = {epoch_loss:.4f}, Training Accuracy = {epoch_accuracy:.4f}")

            # Doğrulama aşaması
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                val_progress = tqdm(enumerate(zip(val_inputs.split(batch_size), val_labels.split(batch_size))), 
                                    total=len(val_inputs) // batch_size, 
                                    desc="Validation", 
                                    leave=True)

                for step, (inputs, labels) in val_progress:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(input_features=inputs, labels=labels)
                    loss = outputs.loss

                    val_loss += loss.item()
                    preds = torch.argmax(outputs.logits, dim=-1)
                    val_correct += (preds == labels.squeeze()).sum().item()
                    val_total += labels.size(0)

                    val_progress.set_postfix({
                        "loss": val_loss / (step + 1),
                        "accuracy": val_correct / val_total
                    })

            val_loss /= len(val_inputs)
            val_accuracy = val_correct / val_total
            print(f"Epoch {epoch + 1}: Val_Loss = {val_loss:.4f}, Val_Accuracy = {val_accuracy:.4f}")

            # Early stopping kontrolü
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if not os.path.exists("best_models"):
                    os.makedirs("best_models")
                torch.save(model.state_dict(), f"best_models/best_model_fold_{fold + 1}.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    # print("Early stopping triggered.")
                    break
        torch.cuda.empty_cache()
        # Eğitim süresini ölçmek için bitiş zamanı
        end_time = time.time()
        duration = end_time - start_time

        print("\nTesting the model on Test Data...")
        # Test aşaması
        model.load_state_dict(torch.load(f"best_models/best_model_fold_{fold + 1}.pt"))  # En iyi modeli yükle
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            test_progress = tqdm(enumerate(zip(test_inputs.split(batch_size), test_labels.split(batch_size))), 
                                total=len(test_inputs) // batch_size, 
                                desc="Test", 
                                leave=True)

            for step, (inputs, labels) in test_progress:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(input_features=inputs, labels=labels)
                loss = outputs.loss

                test_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                test_correct += (preds == labels.squeeze()).sum().item()
                test_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                  # Hata çözümü: Tekil değerler için kontrol
                if labels.squeeze().dim() == 0:
                    all_labels.append(labels.squeeze().cpu().item())  # Tek bir değerse direkt ekle
                else:
                    all_labels.extend(labels.squeeze().cpu().numpy())

                test_progress.set_postfix({
                    "loss": test_loss / (step + 1),
                    "accuracy": test_correct / test_total
                })

        test_loss /= len(test_inputs)
        test_accuracy = test_correct / test_total

        # Sklearn ile metrik hesaplama
        test_accuracy = accuracy_score(all_labels, all_preds)
        test_precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        test_recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        test_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        print(f"Fold {fold + 1} - Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1 Score: {test_f1:.4f}, Duration: {duration:.2f} seconds")

        # Fold metriklerini kaydet
        fold_metrics = {
            "Fold": fold + 1,
            "Accuracy": test_accuracy,
            "Precision": test_precision,
            "Recall": test_recall,
            "F1 Score": test_f1,
            "Duration (s)": duration
        }
        metrics_summary.append(fold_metrics)
        
            # Eski verileri RAM'den silin
        del train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels
        del train_data, val_data, test_data, temp_data
        del model
        torch.cuda.empty_cache()  # GPU belleğini boşaltmak için
        
        gc.collect()

        
    
        
    total_duration += duration
    return pd.DataFrame(metrics_summary),total_duration

# Eğitim ve test çağrısı
# metrics,total_duration = train_openaiwhisper()

# # Metrikleri yazdır
# print(metrics)
