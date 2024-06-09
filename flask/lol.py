from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load and prepare the tokenizer
data = pd.read_csv("dataset/text.csv")
dataframe = pd.DataFrame(data).drop(labels="Unnamed: 0", axis=1)
texts = dataframe["text"]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1

# Find the maximum text length
max_length = max(len(text) for text in texts)

# Define the prediction function
def pred(text):
    emotions = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    tokenized = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(tokenized, maxlen=max_length, padding="post", truncating="post")
    prediction = model.predict(padded)
    emotion = emotions[np.argmax(prediction)]
    return emotion

# Define the routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    predicted_emotion = pred(text)
    return jsonify({'prediction': predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)
