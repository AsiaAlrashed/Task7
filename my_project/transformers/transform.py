import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from mage_ai.data_cleaner.transformer_actions.base import BaseAction
from mage_ai.data_cleaner.transformer_actions.constants import ActionType, Axis
from mage_ai.data_cleaner.transformer_actions.utils import build_transformer_action
from pandas import DataFrame

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


class TextProcessor:
    def __init__(self, vocab_size=5000, max_length=20, oov_token="<OOV>"):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.oov_token = oov_token
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_token)
        self.label_encoder = LabelEncoder()

    def fit_tokenizer(self, texts):
        """
        Tokenizer training on training scripts only
        """
        self.tokenizer.fit_on_texts(texts)

    def transform_texts(self, texts):
        """
        Convert text to number sequence with fill
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding="post")
        return [list(seq) for seq in padded_sequences]  # Convert ndarray to list
        
    def encode_labels(self, labels):
        """
        Encode labels to numbers as int instead of numpy.integer
        """
        return [int(label) for label in self.label_encoder.fit_transform(labels)]


@transformer
def execute_transformer_action(data: tuple, *args, **kwargs) -> tuple:
    """
    Perform text data transformation for both training and test sets using `TextProcessor`
    """
    df_train, df_test = data  # Receive training and test data from Data Loader

    processor = TextProcessor()

    # Train the converter on training texts only.
    processor.fit_tokenizer(df_train["new_tweet_content"])

    # Convert text to number sequence
    df_train["processed_text"] = processor.transform_texts(df_train["new_tweet_content"])
    df_test["processed_text"] = processor.transform_texts(df_test["new_tweet_content"])

    # Convert labels to numbers
    df_train["encoded_label"] = processor.encode_labels(df_train["Label"])
    df_test["encoded_label"] = processor.encode_labels(df_test["Label"])

    return df_train, df_test


@test
def test_output(output, *args) -> None:
    """
    Data validity testing after transformation:
    1. Verify that the output is not empty
    2. Verify that the figures are correct
    3. Verify that the data is numerical and does not contain missing values
    """
    assert output is not None, "Error: Output not found (None)"
    assert isinstance(output, tuple), "Error: Output must be tuple (df_train, df_test)"
    assert len(output) == 2, "Error: Output must contain df_train and df_test"

    df_train, df_test = output

    for df, name in zip([df_train, df_test], ["df_train", "df_test"]):
        assert isinstance(df, DataFrame), f"Error: {name} must be a DataFrame"
        assert "processed_text" in df.columns, f"Error: Column 'processed_text' is missing in {name}!"
        assert "encoded_label" in df.columns, f"Error: Column 'encoded_label' is missing in {name}!"

        # التأكد من أن 'processed_text' عبارة عن قائمة من القوائم
        assert df["processed_text"].apply(lambda x: isinstance(x, list)).all(), f"Error: 'processed_text' must be a list of lists in {name}!"
        
        # التأكد من أن 'encoded_label' يحتوي على أعداد صحيحة
        assert df["encoded_label"].apply(lambda x: isinstance(x, int)).all(), f"Error: 'encoded_label' must contain integers in {name}!"

    print("All tests passed! Data is clean and ready for modeling")
