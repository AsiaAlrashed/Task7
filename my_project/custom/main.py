from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from multiprocessing import Process
import uvicorn

if "custom" not in globals():
    from mage_ai.data_preparation.decorators import custom

# upload the trained model from the path I saved it in
model_path = "E:\\Task7\\lstm_text_classifier.h5"  
model = load_model(model_path)

app = FastAPI()

class TextInput(BaseModel):
    text: str

class TextProcessor:
    def __init__(self, vocab_size=5000, max_length=20, oov_token="<OOV>"):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.oov_token = oov_token
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size, oov_token=self.oov_token
        )

    def fit_tokenizer(self, texts):
        """تدريب الـ Tokenizer على نصوص التدريب فقط"""
        self.tokenizer.fit_on_texts(texts)

    def transform_texts(self, texts):
        """تحويل النصوص إلى تسلسل أرقام مع التعبئة"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_length, padding="post")


processor = TextProcessor()

@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}

@app.post("/predict/")
def predict(text_input: TextInput):
    # Convert input text to sequence
    text = [text_input.text]
    processed_text = processor.transform_texts(text)

    # Predicting using the model
    prediction = model.predict(np.array(processed_text))

    # Return prediction
    return {"prediction": float(prediction[0][0])}


#Running FastAPI as the last step in the Mage pipeline
@custom
def run_api(*args, **kwargs):
    uvicorn.run(app, host="0.0.0.0", port=8000)
