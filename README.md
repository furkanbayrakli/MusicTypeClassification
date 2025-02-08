# MusicTypeClassification

        In this project, music genre classification has been performed using signal processing
        and machine learning techniques.
        
        Recently, an increase in music data and the emergence of various music genres have
        been observed. Music genre classification plays a critical role in organizing and
        accurately categorizing this vast amount of music data.
        
        In this study, the GTZAN dataset, which contains data from different music genres,
        has been used. The GTZAN dataset includes 10 different music genres (rock, jazz,
        classical, pop, blues, metal, reggae, hip-hop, country, and disco), with 100 audio tracks
        of 30 seconds each per genre.
        
        During the data processing stage, Short-Time Fourier Transform (STFT) was initially
        applied to samples selected from the music genres. The matrix obtained from the STFT
        is an essential tool for audio analysis. Subsequently, these samples were converted into
        spectrograms and mel spectrograms, visualizing both frequency and time information.
        Finally, specific Convolutional Neural Network (CNN)-based models were used in the
        classification stage.
        
        Keywords: Music Genre, Classification, GTZAN, Short-Time Fourier Transform,
        Audio Analysis, Spectrogram, Mel Spectrogram, Convolutional Neural Networks

# Model Comparison and Performance Evaluation Using K-Fold Cross-Validation
        In this project, the goal is to compare five different CNN-based models used for music genre classification in terms of performance metrics such as accuracy, precision, recall, and F1 score. To achieve this, K-fold cross-validation is also used, allowing the models' performance to be tested on different train and test datasets. The specified models have been trained and tested, with the aim of determining the model that demonstrates the highest performance.
# CNN Models Used 
        InceptionV3
        DenseNet121 
        Openai/whisper-tiny 
        MusicCNN
        Speechbrain/google-speech-commands
# Steps for running codes

        This project uses the extractMusicnnFeatures.py file, which relies on the musicnn library and is compatible only with Python 3.7. Therefore, follow the steps below to run the project:
        
        Set Up Python 3.7 Environment:
        First, you need to install the required libraries in the requirementsMusicnn.txt.
        
        Download the Dataset:(https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
        Download the dataset  and copy the archive folder from it into the directory where extractMusicnnFeatures.py is located.
        
        Run the extractMusicnnFeatures.py Script:
        Execute this script to generate a .csv file. You will need this file for the next steps.
        
        Switch to Python 3.9 Environment:
        For the remaining parts of the code, you should use an environment with Python 3.9. In this environment, install the necessary libraries (as specified in the requirements.txt), then place .csv file you generated earlier and all the .py files except extractMusicnnFeatures.py  in the directory.
        
        Copy the Dataset:
        Again, download the dataset from the database folder and copy the archive folder into the directory containing the project files.
        
        Generate Spectrogram and MelSpectrogram:
        To begin, run the spectrogram.py script to generate the spectrogram and melspectrogram files.
        
        Data Augmentation (Optional):
        If you want to expand the dataset, you can run the augmentation.py script. This step is optional, but it can help improve the accuracy of the model.
        
        Compile the Model and View Results:
        Finally, after completing all the previous steps, you can compile the model by running the main.py script. This will allow you to view the results.

