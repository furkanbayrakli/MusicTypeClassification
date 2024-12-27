import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import numpy as np
import pandas as pd
import time
# Generic path and parameters
mel_spectrogram_path = "mel_spectrograms"
spectrogram_path = "spectrograms"
image_size = (224, 224)
batch_size = 32
num_folds = 5

# Create a function to build the model using InceptionV3
def build_inceptionv3_model(input_shape=(299, 299, 3)):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)  # 10 sınıf için çıkış katmanı
    model = Model(inputs=base_model.input, outputs=predictions)

    # Transfer öğrenme için temel katmanları dondur
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train and evaluate models
def train_and_evaluate(data_path, description):
    # Veri hazırlama
    filepaths = []
    labels = []
    for label in os.listdir(data_path):
        class_path = os.path.join(data_path, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                filepaths.append(os.path.join(class_path, file))
                labels.append(label)

    data = pd.DataFrame({'filepath': filepaths, 'label': labels})
    label_map = {label: idx for idx, label in enumerate(data['label'].unique())}
    data['label_encoded'] = data['label'].map(label_map)

    
    datagen = ImageDataGenerator(rescale=1./255)
   
    metrics_summary = []
    total_duration=0
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(data['filepath'], data['label_encoded'])):
        print(f"\nModel InceptionV3 - Training on {description} - at Fold {fold + 1}/{num_folds}")
        
        test_data = data.iloc[test_idx]
        temp_data = data.iloc[train_idx]
        train_data, val_data = train_test_split(temp_data, test_size=0.1, stratify=temp_data['label_encoded'], random_state=42)
        
        # Veri jeneratörleri
        test_gen = datagen.flow_from_dataframe(
            test_data,
            x_col='filepath',
            y_col='label',
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        train_gen = datagen.flow_from_dataframe(
            train_data,
            x_col='filepath',
            y_col='label',
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        val_gen = datagen.flow_from_dataframe(
            val_data,
            x_col='filepath',
            y_col='label',
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )

        # log_data_split(train_data, val_data, test_data, fold + 1)

        # Model oluştur ve eğit
        model = build_inceptionv3_model(input_shape=(image_size[0], image_size[1], 3))
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        # Eğitim süresini ölç
        start_time = time.time()
        model.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=[callback], verbose=1)
        end_time = time.time()
        duration = end_time - start_time
        total_duration += duration
        
        # Test performansı
        print(f"Testing on {description}")
        test_labels = test_gen.classes
        test_preds = model.predict(test_gen, verbose=1)
        test_preds = np.argmax(test_preds, axis=1)

        accuracy = accuracy_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds, average='weighted')
        recall = recall_score(test_labels, test_preds, average='weighted')
        f1 = f1_score(test_labels, test_preds, average='weighted')
        
        metrics_summary.append({
            'Fold': fold + 1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Duration (s)': duration
        })
        print(metrics_summary)

    print(f"\nTotal Duration for {description}: {total_duration:.2f} seconds")
    return pd.DataFrame(metrics_summary),total_duration


def train_inceptionV3():
    mel_metrics,mel_total_time = train_and_evaluate(mel_spectrogram_path, "mel spectrograms")
    spec_metrics,spec_total_time = train_and_evaluate(spectrogram_path, "regular spectrograms")
    return {
        "mel": mel_metrics,
        "spec": spec_metrics
    }, mel_total_time + spec_total_time

def log_data_split(train_data, val_data, test_data, fold_number, log_file="seperatedDataInception.txt"):
    with open(log_file, "a") as f:
        f.write(f"Fold {fold_number}\n")
        f.write("Train Data:\n")
        f.writelines([f"{row}\n" for row in train_data['filepath']])
        f.write("\nValidation Data:\n")
        f.writelines([f"{row}\n" for row in val_data['filepath']])
        f.write("\nTest Data:\n")
        f.writelines([f"{row}\n" for row in test_data['filepath']])
        f.write("\n" + "="*50 + "\n")

# # Train and evaluate models separately for mel and regular spectrograms
# # print(f"\nComparison:")

# mel_metrics = train_and_evaluate(mel_spectrogram_path, "mel spectrograms")
# # print(f"Inception Mel Spectrogram Metrics:\n{mel_metrics}")

# spec_metrics = train_and_evaluate(spectrogram_path, "regular spectrograms")
# print(f"Inception Regular Spectrogram Metrics:\n{spec_metrics}")











