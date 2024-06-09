
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
import gradio

model.save('model.h5')