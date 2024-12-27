import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def spec_augment(image, freq_mask_param=15, time_mask_param=15):
    """
    SpecAugment uygulanmış bir görüntü döndürür.
    :param image: Girdi görüntüsü (spectrogram)
    :param freq_mask_param: Frekans ekseninde mask parametresi
    :param time_mask_param: Zaman ekseninde mask parametresi
    """
    augmented_image = image.copy()
    num_freq_channels = augmented_image.shape[0]
    num_time_steps = augmented_image.shape[1]

    # Frekans Maskesi
    freq_mask = random.randint(0, freq_mask_param)
    f0 = random.randint(0, num_freq_channels - freq_mask)
    augmented_image[f0:f0 + freq_mask, :] = 0

    # Zaman Maskesi
    time_mask = random.randint(0, time_mask_param)
    t0 = random.randint(0, num_time_steps - time_mask)
    augmented_image[:, t0:t0 + time_mask] = 0

    return augmented_image

def apply_spec_augmentation(out_dir):
    genres = os.listdir(out_dir)
    print(f"{out_dir}:")
    for genre in genres:
        genre_path = os.path.join(out_dir, genre)
        files = [f for f in os.listdir(genre_path) if f.endswith('.png')]
        total_files = len(files)

        # Mevcut en yüksek sayıyı bulma
        max_index = max([int(f.split('.')[1].split('_')[0]) for f in files if 'aug' not in f and '.' in f], default=0)

        for i, file in enumerate(files):
            file_path = os.path.join(genre_path, file)
            if '_aug' in file:  # Augment edilmiş dosyaları atla
                continue

            # Görüntüyü yükleme ve augment uygulama
            original_image = np.array(Image.open(file_path))
            augmented_image = spec_augment(original_image)

            # Yeni dosya adını belirleme
            if "mel" in file:
                new_filename = f"{genre}.{max_index + i + 1:05d}_mel_aug.png"
            else:
                new_filename = f"{genre}.{max_index + i + 1:05d}_spectrogram_aug.png"
            new_file_path = os.path.join(genre_path, new_filename)

            # Augmented görüntüyü renkli olarak kaydetme
            plt.imsave(new_file_path, augmented_image)

            # Yüzdelik ilerlemeyi gösterme
            progress = (i + 1) / total_files * 100
            print(f"\r{genre} : {progress:.2f}%", end="")
        print()  # Her türün sonunda yeni bir satıra geçiş

apply_spec_augmentation("mel_spectrograms")
apply_spec_augmentation("spectrograms")
