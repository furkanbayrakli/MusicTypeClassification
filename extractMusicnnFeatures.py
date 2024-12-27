import os
import pandas as pd
from musicnn.extractor import extractor

def extract_features_to_csv(data_path, genres, output_file='music_features.csv'):
    data = []
    for genre in genres:
        genre_path = os.path.join(data_path, genre)
        for file in os.listdir(genre_path):
            if file.endswith('.wav'):
                file_path = os.path.join(genre_path, file)
                print(file_path)
                try:
                    # Özellik çıkarma
                    features, tags, additional_info = extractor(file_path)
                    
                    # Her bir dosyanın özelliklerini listeye ekle
                    feature_row = features.flatten().tolist()  # Özellikleri düzleştir
                    feature_row.append(genre)  # Tür bilgisini ekle
                    feature_row.append(file)  # Dosya adını ekle
                    data.append(feature_row)
                    
                except Exception as e:
                    print(f"Error with file {file_path}: {e}")
    
    # DataFrame oluşturma
    feature_columns = [f"feature_{i}" for i in range(len(data[0]) - 2)]  # Tür ve dosya adı sütunları hariç
    columns = feature_columns + ['genre', 'file_name']
    df = pd.DataFrame(data, columns=columns)
    
    # CSV dosyasına kaydetme
    df.to_csv(output_file, index=False)
    print(f"Features extracted and saved to '{output_file}'")

# Veri yolu ve türler
data_path = r'archive\Data\genres_original'
genres = os.listdir(data_path)

# Özellikleri çıkar ve CSV'ye kaydet
extract_features_to_csv(data_path, genres)

print("Features extracted and saved to 'music_features.csv'")


#  df =read(csv) python sınıflandırma algoritmaları
#  xgboost catgbm 