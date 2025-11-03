import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---- Config ----
MODEL_PATH = 'movie_reviews_rnn.h5' # or path to saved_model directory
TOKENIZER_PATH = 'tokenizer.pickle'
MAX_LEN = 200


# ---- Utilities ----
@st.cache_resource
def load_model_and_tokenizer(model_path=MODEL_PATH, tokenizer_path=TOKENIZER_PATH):
    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
        return model, tokenizer


def clean_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


# ---- Streamlit UI ----
st.title('Movie Review Sentiment (RNN)')
st.write('Type or paste a movie review and click Predict.')


model, tokenizer = load_model_and_tokenizer()


review = st.text_area('Review', height=100)
if st.button('Predict'):
    if not review.strip():
        st.warning('Please enter a review.')
    else:
        cleaned = clean_text(review)
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        prob = model.predict(padded)[0][0]
        st.write(f'**Positive probability:** {prob:.3f}')
        label = 'Positive' if prob > 0.5 else 'Negative'
        st.markdown(f'### Prediction: {label}')
        st.info('Tip: this model is a demonstration — accuracy depends on training and hyperparameters.')