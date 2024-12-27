import pandas as pd
from inceptionV3 import train_inceptionV3
from denseNet121 import train_denseNet
from musiCNNmodel import train_musicnn
from openaiWhisperModel import train_openaiwhisper
# Fold sonuçlarını birleştirmek için fonksiyon
def combine_kfold_results(model_name, dataset_name, metrics_list):
    combined_results = []
    
    for fold, metrics in enumerate(metrics_list, start=1):
        current_fold = 1 if "Classifier" in metrics else fold
        model_name = metrics.get("Classifier", model_name)
        combined_results.append([
            model_name,
            dataset_name,
            current_fold,
            metrics["Accuracy"],
            metrics["Precision"],
            metrics["Recall"],
            metrics["F1 Score"],
            metrics["Duration (s)"]
        ])
    return combined_results


# Ortalama metrikleri hesaplama
def calculate_mean_results(metrics_df):
    mean_results = metrics_df.groupby(["Model", "Dataset"]).mean(numeric_only=True).reset_index()
    mean_results["Fold"] = "Mean"  # Fold olarak "Mean" eklenir
    mean_results = mean_results.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)  # Accuracy değerine göre sırala
    return mean_results


# Inception, DenseNet ve Musicnn modellerini eğit

inception_metrics, inception_total_time = train_inceptionV3()
dense_metrics, dense_total_time = train_denseNet()
whisper_metrics, whisper_total_time = train_openaiwhisper()
musicnn_metrics, musicnn_total_time = train_musicnn()

metrics_data = []
metrics_data += combine_kfold_results("InceptionV3", "Mel Spectrogram", inception_metrics["mel"].to_dict(orient="records"))
metrics_data += combine_kfold_results("InceptionV3", "Regular Spectrogram", inception_metrics["spec"].to_dict(orient="records"))
metrics_data += combine_kfold_results("DenseNet121", "Mel Spectrogram", dense_metrics["mel"].to_dict(orient="records"))
metrics_data += combine_kfold_results("DenseNet121", "Regular Spectrogram", dense_metrics["spec"].to_dict(orient="records"))
metrics_data += combine_kfold_results("OpenaiWhisper", "Audio", whisper_metrics.to_dict(orient="records"))
metrics_data += combine_kfold_results("Musicnn", "Musicnn_features.csv", musicnn_metrics.to_dict(orient="records"))

# DataFrame'e dönüştür
metrics_df = pd.DataFrame(metrics_data, columns=["Model", "Dataset", "Fold", "Accuracy", "Precision", "Recall", "F1 Score", "Duration (s)"])

# Ortalama metrikler için DataFrame
mean_metrics_df = calculate_mean_results(metrics_df)

# Tabloyu yazdır
print("\nModel ve Fold Karşılaştırma Tablosu:")
print(metrics_df)

print("\nModel ve Fold Ortalamaları:")
print(mean_metrics_df)

# Toplam süreleri yazdır
print(f"\nTotal Duration for InceptionV3: {inception_total_time:.2f} seconds")
print(f"Total Duration for DenseNet121: {dense_total_time:.2f} seconds")
print(f"Total Duration for Openai Whisper: {whisper_total_time:.2f} seconds")
print(f"Total Duration for Musicnn: {musicnn_total_time:.2f} seconds")
