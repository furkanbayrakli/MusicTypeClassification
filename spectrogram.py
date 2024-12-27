import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def create_spectrogram(archive_path):
    # Çıktı klasörleri
    mel_output_dir = "mel_spectrograms"
    spectrogram_output_dir = "spectrograms"

    # Çıktı klasörlerini oluşturma
    os.makedirs(mel_output_dir, exist_ok=True)
    os.makedirs(spectrogram_output_dir, exist_ok=True)

    # Hata nedeniyle atlanan dosyalar için liste
    skipped_files = []

    # Tüm türleri ve dosyaları işleme
    for genre in os.listdir(archive_path):
        genre_path = os.path.join(archive_path, genre)
        
        if os.path.isdir(genre_path):
            print(f"Processing genre: {genre}")

            # Tür için ayrı mel ve spectrogram klasörleri oluşturma
            genre_mel_dir = os.path.join(mel_output_dir, genre)
            genre_spec_dir = os.path.join(spectrogram_output_dir, genre)
            
            os.makedirs(genre_mel_dir, exist_ok=True)
            os.makedirs(genre_spec_dir, exist_ok=True)

            audio_files = [f for f in os.listdir(genre_path) if f.endswith(".wav")]
            total_files = len(audio_files)

            for i, audio_file in enumerate(audio_files):
                audio_path = os.path.join(genre_path, audio_file)

                try:
                    # Ses dosyasını yükleme
                    y, sr = librosa.load(audio_path, sr=None)

                    # Mel Spectrogram oluşturma ve kaydetme
                    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

                    plt.figure(figsize=(6, 4))
                    librosa.display.specshow(mel_spectrogram_db, sr=sr, x_axis=None, y_axis=None)
                    plt.axis('off')
                    mel_file_path = os.path.join(genre_mel_dir, f"{os.path.splitext(audio_file)[0]}_mel.png")
                    plt.savefig(mel_file_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    # Normal Spectrogram oluşturma ve kaydetme
                    stft = librosa.stft(y)
                    spectrogram_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

                    plt.figure(figsize=(6, 4))
                    librosa.display.specshow(spectrogram_db, sr=sr, x_axis=None, y_axis=None)
                    plt.axis('off')
                    spectrogram_file_path = os.path.join(genre_spec_dir, f"{os.path.splitext(audio_file)[0]}_spectrogram.png")
                    plt.savefig(spectrogram_file_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                except Exception as e:
                    print(f"\nSkipping {audio_file} due to error: {e}")
                    skipped_files.append(audio_file)
                    continue

                progress = (i + 1) / total_files * 100
                print(f"\r{genre}: {progress:.2f}%", end="")

            print()

    if skipped_files:
        print("\nSkipped files:")
        for file in skipped_files:
            print(f"- {file}")
    else:
        print("\nAll files processed successfully.")

    print("Processing complete!")
    

# Archive klasörünün yolunu belirtin
archive_path = "archive/Data/genres_original"

create_spectrogram(archive_path)