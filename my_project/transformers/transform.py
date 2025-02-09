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
        """تدريب الـ Tokenizer على نصوص التدريب فقط"""
        self.tokenizer.fit_on_texts(texts)

    def transform_texts(self, texts):
        """تحويل النصوص إلى تسلسل أرقام مع التعبئة"""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length, padding="post")
        return [list(seq) for seq in padded_sequences]  # تحويل ndarray إلى list لتجنب الخطأ

    def encode_labels(self, labels):
        """ترميز التسميات إلى أرقام كـ int بدلاً من numpy.integer"""
        return [int(label) for label in self.label_encoder.fit_transform(labels)]


@transformer
def execute_transformer_action(data: tuple, *args, **kwargs) -> tuple:
    """
    تنفيذ تحويل البيانات النصية لكلا مجموعتي التدريب والاختبار باستخدام `TextProcessor`
    """
    df_train, df_test = data  # استلام بيانات التدريب والاختبار من Data Loader

    processor = TextProcessor()

    # تدريب المحول على نصوص التدريب فقط
    processor.fit_tokenizer(df_train["new_tweet_content"])

    # تحويل النصوص إلى تسلسل أرقام
    df_train["processed_text"] = processor.transform_texts(df_train["new_tweet_content"])
    df_test["processed_text"] = processor.transform_texts(df_test["new_tweet_content"])

    # تحويل التسميات إلى أرقام
    df_train["encoded_label"] = processor.encode_labels(df_train["Label"])
    df_test["encoded_label"] = processor.encode_labels(df_test["Label"])

    return df_train, df_test


@test
def test_output(output, *args) -> None:
    """
    اختبار صحة البيانات بعد التحويل:
    1. التحقق من أن الإخراج ليس فارغًا
    2. التحقق من صحة الأشكال
    3. التأكد من أن البيانات عددية ولا تحتوي على قيم مفقودة
    """
    assert output is not None, "❌ خطأ: الإخراج غير موجود (None)"
    assert isinstance(output, tuple), "❌ خطأ: يجب أن يكون الإخراج Tuple (df_train, df_test)"
    assert len(output) == 2, "❌ خطأ: الإخراج يجب أن يحتوي على df_train و df_test"

    df_train, df_test = output

    for df, name in zip([df_train, df_test], ["df_train", "df_test"]):
        assert isinstance(df, DataFrame), f"❌ خطأ: {name} يجب أن يكون DataFrame"
        assert "processed_text" in df.columns, f"❌ خطأ: العمود 'processed_text' مفقود في {name}!"
        assert "encoded_label" in df.columns, f"❌ خطأ: العمود 'encoded_label' مفقود في {name}!"

        # التأكد من أن 'processed_text' عبارة عن قائمة من القوائم
        assert df["processed_text"].apply(lambda x: isinstance(x, list)).all(), f"❌ خطأ: 'processed_text' يجب أن يكون قائمة من القوائم في {name}!"
        
        # التأكد من أن 'encoded_label' يحتوي على أعداد صحيحة
        assert df["encoded_label"].apply(lambda x: isinstance(x, int)).all(), f"❌ خطأ: 'encoded_label' يجب أن يحتوي على أعداد صحيحة في {name}!"

    print("✅ جميع الاختبارات نجحت! البيانات نظيفة وجاهزة للنموذج 🚀")
