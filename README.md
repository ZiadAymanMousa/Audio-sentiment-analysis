# Audio-sentiment-analysis

building an audio sentiment analysis model using various libraries and techniques in Python. The primary objective is to classify audio files into different sentiment categories. The project employs data preprocessing, feature extraction, data augmentation, and deep learning model training to achieve this goal.

Libraries and Dependencies
The project uses a range of libraries:

  os, random, numpy, pandas: For general-purpose operations and data manipulation.
  keras: For building and training the neural network model.
  librosa: For audio processing and feature extraction.
  matplotlib, seaborn: For data visualization.
  sklearn: For data splitting and evaluation metrics.
  IPython.display: For playing audio files.
  scipy: For signal processing.
  tqdm: For progress bars during iterations.
  
 Data Preparation
  Mounting Google Drive: The dataset is stored in Google Drive, and it's mounted to the Colab environment for easy access.
  Loading Dataset: The training dataset is loaded from a CSV file which contains filenames and their corresponding sentiment labels.
  Visualizing Class Distribution: A pie chart is plotted to show the distribution of different sentiment classes in the training data.
  Audio Processing and Feature Extraction
  Loading Audio Files: Audio files are loaded using librosa.
  Spectrogram Generation: Spectrograms and Mel Power Spectrograms are generated for visualizing the audio signals.
  MFCC Calculation: Mel-Frequency Cepstral Coefficients (MFCCs) are extracted as features from the audio signals.
  Trimming Silence: Silence in audio files is trimmed to focus on the relevant parts of the signal.
  
Data Augmentation
  
  To increase the diversity of the training data, several augmentation techniques are applied:
  Adding Noise: White noise is added to the audio signals.
  Shifting: Audio signals are randomly shifted.
  Stretching: Audio signals are stretched.
  Pitch Tuning: The pitch of the audio signals is modified.
  Dynamic Range Change: The dynamic range of the audio signals is altered.

  
Model Training
  Model Architecture: A Convolutional Neural Network (CNN) is built with several Conv1D layers, Batch Normalization, Dropout layers, MaxPooling, and Dense layers. The model is compiled with the SGD optimizer and categorical cross-entropy     loss.
  Callbacks: Various callbacks are used during training:
  ReduceLROnPlateau: Reduces learning rate when a metric has stopped improving.
  ModelCheckpoint: Saves the model with the best validation loss.
  Training: The model is trained on the augmented dataset with a validation split, and the training history is plotted.
  Evaluation Metrics
  Custom evaluation metrics including precision, recall, and F1-score are defined using Keras backend functions.
