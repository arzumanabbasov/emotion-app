# Emotion Prediction App

This project is a web application that predicts the emotion of a given text input using a pre-trained Bidirectional Long Short-Term Memory (BiLSTM) model. The application is built using Flask for the backend and Tailwind CSS for the frontend.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model and Tokenizer](#model-and-tokenizer)
- [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Python 3.7+
- Flask
- TensorFlow
- Keras
- Pandas
- NumPy

### Steps

1. Clone the repository:
    ```bash
    git clone https://github.com/arzumanabbasov/emotion-app.git
    cd emotion-app
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Place the pre-trained model (`bidirectionallstmmodel.h5`) and tokenizer (`bilstm-tokenizer.pickle`) files in the project root directory.

5. Run the Flask app:
    ```bash
    python app.py
    ```

6. Open your web browser and navigate to `http://localhost:5000`.

## Usage

1. Open the web application.
2. Enter the text you want to analyze in the provided text area.
3. Click the "Predict Emotion" button.
4. The predicted emotion will be displayed on the screen.

## Project Structure

```
emotion-app/
│
├── app.py                   # Main Flask application
├── bilstm.py                # Model loading and prediction logic
├── templates/
│   └── index.html           # HTML template for the web application
├── bidirectionallstmmodel.h5 # Pre-trained BiLSTM model (not included in repo)
├── bilstm-tokenizer.pickle  # Tokenizer for preprocessing (not included in repo)
├── requirements.txt         # Python dependencies
└── README.md                # Project README file
```

## Model and Tokenizer

- **Model:** The BiLSTM model (`bidirectionallstmmodel.h5`) is pre-trained to classify emotions in text.
- **Tokenizer:** The tokenizer (`bilstm-tokenizer.pickle`) is used to preprocess and convert text into sequences that the model can understand.

## Prediction

The core prediction logic is handled in the `bilstm.py` file. Here is a brief overview of the process:

1. **Loading the Model and Tokenizer:**
    ```python
    def load_model_and_tokenizer(model_path, tokenizer_path):
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        
        return model, tokenizer
    ```

2. **Preprocessing Text:**
    ```python
    def preprocess_texts(texts, tokenizer, max_sequence_length):
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
        return padded_sequences
    ```

3. **Making Predictions:**
    ```python
    def predict_emotion(text):
        numerical_prediction = predict(text, model, tokenizer)
        emotion_prediction = emotion_labels.get(numerical_prediction, "Unknown")
        return emotion_prediction
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License.
