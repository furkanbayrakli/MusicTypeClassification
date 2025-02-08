# MusicTypeClassification
Steps for running codes:

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

