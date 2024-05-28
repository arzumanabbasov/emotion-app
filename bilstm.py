import pandas as pd
import numpy as np
import pickle
import tensorflow
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Define paths to the model and tokenizer
model_path = "bidirectionallstmmodel.h5"
tokenizer_path = "bilstm-tokenizer.pickle"


def load_model_and_tokenizer(model_path, tokenizer_path):
    """
    Load the pre-trained BiLSTM model and tokenizer.
    """
    # Load the pre-trained model
    model = load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Load the tokenizer
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)

    return model, tokenizer


def preprocess_texts(texts, tokenizer, max_sequence_length):
    """
    Preprocess texts by tokenizing and padding.
    """
    # Tokenize the texts
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences to the same length
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    return padded_sequences


def predict(text, model, tokenizer, max_sequence_length=100):
    """
    Predict the emotion of a given text using the BiLSTM model.
    """
    # Preprocess the text
    processed_text = preprocess_texts([text], tokenizer, max_sequence_length)

    # Predict the emotion
    prediction = model.predict(processed_text)

    # Get the emotion with the highest probability
    predicted_emotion = np.argmax(prediction, axis=1)[0]

    return predicted_emotion


# Load the model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

# Define the mapping of numerical labels to emotion text labels
emotion_labels = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


def predict_emotion(text):
    """
    Predict the emotion of a given text using the BiLSTM model.

    Args:
    text (str): The input text to predict the emotion for.

    Returns:
    str: The predicted emotion as a text label.
    """
    # Get the numerical prediction from the model
    numerical_prediction = predict(text, model, tokenizer)

    # Map the numerical prediction to the corresponding emotion label
    emotion_prediction = emotion_labels.get(numerical_prediction, "Unknown")

    return emotion_prediction

