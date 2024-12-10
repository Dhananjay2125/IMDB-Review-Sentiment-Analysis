
# IMDB Review Sentiment Analysis

This project demonstrates sentiment analysis on the IMDB movie review dataset using deep learning techniques. The goal is to classify reviews as positive or negative sentiments using a Long Short-Term Memory (LSTM) model.
## Project Steps

### 1. Importing Dataset from Kaggle
The dataset is sourced from Kaggle's IMDB movie review dataset. It contains labeled reviews categorized as positive or negative. The dataset was downloaded and extracted for further processing.

### 2. Extracting Files
After downloading, the dataset was extracted and organized to separate reviews and their corresponding sentiment labels.

### 3. Encoding Labels
The sentiment labels were encoded:

- Positive reviews as 1
- Negative reviews as 0

### 4. Train-Test Split
The data was divided into:

- Training set: Used to train the model.
- Test set: Used to evaluate model performance.

### 5. Tokenizing the Data
Each review was tokenized, converting the text into numerical sequences to feed into the neural network. The Tokenizer class from the TensorFlow library was used for this step.

### 6. Applying Padding
To ensure uniform input lengths, padding was applied to the tokenized data using pad_sequences from TensorFlow. This ensures consistency in the LSTM model's input shape.

### 7. Setting up the LSTM Model
An LSTM-based deep learning model was designed with the following layers:

- Embedding layer: To handle word embeddings.
- LSTM layer: To capture sequential dependencies.
- Dense layer: For final binary classification.

### 8. Training the Model
The model was trained using the training set with the following parameters:

- Epochs: 5
- Loss function: Binary cross-entropy
- Optimizer: Adam
- Metrics: Accuracy

### 9. Building a Predictive System
A predictive system was implemented to take new reviews as input and output the predicted sentiment:

- Positive for outputs closer to 1
- Negative for outputs closer to 0


## Technologies and Tools

- Python
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib/Seaborn (for visualizations)
- Kaggle API (for dataset)