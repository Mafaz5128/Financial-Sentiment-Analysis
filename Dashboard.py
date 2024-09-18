import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your trained model, tokenizer, and label encoder
from tensorflow.keras.models import load_model
import joblib

# Load model, tokenizer, and label encoder (ensure these are saved after training)
model = load_model('sentiment_model.h5')  # Replace with your model path
tokenizer = joblib.load('tokenizer.pkl') # Load your tokenizer (you might save it with pickle)
le = joblib.load('label_encoder.pkl') # Load your label encoder (you might save it with pickle)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit app layout
st.title("Sentiment Analysis Dashboard")
st.write("Enter text to analyze its sentiment.")

# Text input
user_input = st.text_area("Text Input")

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the input text
        processed_text = preprocess(user_input)
        
        # Convert to sequence and pad
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=your_max_length, padding='post')
        
        # Make prediction
        prediction = model.predict(padded_sequence)
        predicted_class = np.argmax(prediction, axis=1)
        
        # Get the predicted label
        sentiment = le.classes_[predicted_class[0]]
        
        # Display result
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter some text to analyze.")
