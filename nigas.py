import tensorflow as tf
import numpy as np
import pandas as pd
import tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from keras.regularizers import l2
import pickle
from sklearn.utils import class_weight
import gradio as gr
#files.upload()


#!mkdir ~/.kaggle/
#!mv kaggle.json ~/.kaggle/
#!kaggle datasets download -d nelgiriyewithana/emotions
#!unzip emotions.zip

#Preparing the Dataset Dataframe
data= pd.read_csv("dataset/text.csv")
dataframe=pd.DataFrame(data)
dataframe = dataframe.drop(labels="Unnamed: 0", axis=1)

#Balancing number of samples
dataframe2 = dataframe[(dataframe['label'] == 0) | (dataframe['label'] == 1)]
dataframe2 = dataframe2.drop(dataframe2.sample(frac=0.7, random_state=42).index)


dataframe4 = dataframe[(dataframe['label'] == 4)]
dataframe4 = dataframe4.drop(dataframe4.sample(frac=0.11, random_state=42).index)


dataframe3 = dataframe[(dataframe['label'] == 3)]
dataframe3 = dataframe3.drop(dataframe3.sample(frac=0.26, random_state=42).index)


dataframe5 = pd.concat([dataframe3, dataframe4])


dataframe = dataframe[(dataframe['label'] != 0) & (dataframe['label'] != 1) & (dataframe['label'] != 3) & (dataframe['label'] != 4)]
dataframereal = pd.concat([dataframe, dataframe2, dataframe5])



removed = "i "
removed2 = "feel"
dataframev = dataframereal["text"].str.replace(removed, " ")
dataframerealest = dataframev.str.replace(removed2, " ")
dataframerealest = pd.concat([dataframerealest, dataframereal["label"]], axis=1)
df = pd.DataFrame(dataframerealest)
df

#Tokenising text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequences = tokenizer.texts_to_sequences(df["text"])
vocab_size = len(tokenizer.word_index) + 1

#Finding the maximum text length
max_length = 0
for text in df["text"]:
  if len(text) > max_length:
    max_length = len(text)

max= len(text)

#Padding
pad = pad_sequences(sequences, maxlen=max, padding="post", truncating="post")

#Splitting testing and training data
x= df["text"]
y= df["label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, shuffle=True, random_state=42)
x_train_tokenized = tokenizer.texts_to_sequences(x_train)
x_test_tokenized = tokenizer.texts_to_sequences(x_test)
x_train_padded = pad_sequences(x_train_tokenized, maxlen=max, padding='post')
x_test_padded = pad_sequences(x_test_tokenized, maxlen=max, padding='post')

#Building the model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=100, input_length=max, mask_zero=True, embeddings_initializer="glorot_uniform"),
    Bidirectional(LSTM(128, dropout=0.3,return_sequences=True, kernel_regularizer=l2(0.001))),
    Bidirectional(LSTM(64, dropout=0.2,  return_sequences=True)),
    tf.keras.layers.GlobalAveragePooling1D(),
    Dense(6, activation="softmax")
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_padded, y_train, epochs=2, batch_size=32, validation_data=(x_test_padded, y_test))

model.save('model.h5')

def pred(text):
  emotions={0:"sadness",
           1: "joy",
           2:"love",
           3:"anger",
           4:"fear",
           5:"surprise"}
  tokenized = tokenizer.texts_to_sequences([text])
  padded = pad_sequences(tokenized, maxlen=max, padding= "post", truncating="post")
  prediction = model.predict(padded)
  emotion = emotions[np.argmax(prediction)]
  print(emotion)
  return emotion

#interface = gr.Interface(
    #fn=pred,
    #inputs= gr.Textbox(lines=5, placeholder="Enter your text"),
    #outputs="text"
#)
#interface.launch()