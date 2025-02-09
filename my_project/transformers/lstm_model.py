import tensorflow
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from my_project.ml_flow import MLFlow, MLflowCallback

if "transformer" not in globals():
    from mage_ai.data_preparation.decorators import transformer
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def train_lstm_model(data: tuple, *args, **kwargs) -> str:
    """
    تدريب نموذج LSTM على البيانات النصية.

    Args:
    - data (tuple): يتكون من `df_train` و `df_test`، يحتوي كل منهما على:
        * `processed_text`: قائمة من القوائم تحتوي على النصوص المرمّزة.
        * `encoded_label`: التصنيفات المحولة إلى أرقام.

    Returns:
    - str: مسار حفظ النموذج المدرب.
    """

    df_train, df_test = data  # تفريغ البيانات المحولة

    # تحويل البيانات إلى NumPy arrays
    X_train = np.array(df_train["processed_text"].tolist())
    y_train = np.array(df_train["encoded_label"].tolist())

    X_test = np.array(df_test["processed_text"].tolist())
    y_test = np.array(df_test["encoded_label"].tolist())

    # تقسيم بيانات التدريب إلى تدريب وتحقق
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # إعداد متغيرات النموذج
    vocab_size = 5000  # نفس الحجم المستخدم في `TextProcessor`
    embedding_dim = 64
    max_length = X_train.shape[1]

    # إنشاء نموذج LSTM
    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            LSTM(64, return_sequences=True),
            Dropout(0.5),
            LSTM(32),
            Dropout(0.5),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),  # للخروج ثنائي
        ]
    )

    # تجميع النموذج
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    # إعداد MLflow
    mlflow_tracker = MLFlow(experiment_name="LSTM_Text_Classification")
    mlflow_callback = MLflowCallback(mlflow_tracker)

    # تدريب النموذج
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[mlflow_callback],  # إضافة MLflow كـ Callback
    )

    # تقييم النموذج على بيانات الاختبار
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"✅ دقة النموذج على بيانات الاختبار: {accuracy:.4f}")

    # حفظ النموذج
    model_path = "E:\\Task7\\lstm_text_classifier.h5"
    model.save(model_path)
    print(f"✅ تم حفظ النموذج في: {model_path}")

    # تسجيل النموذج في MLflow
    mlflow_tracker.log_model(model, "lstm_text_classifier")
    mlflow_tracker.end_run()

    return model_path


@test
def test_output(output, *args) -> None:
    """
    اختبار صحة المخرجات.
    - التأكد من أن مسار النموذج الناتج صحيح.
    """
    assert isinstance(
        output, str
    ), "❌ خطأ: الإخراج يجب أن يكون نصًا يحتوي على مسار النموذج"
    assert output.endswith(".h5"), "❌ خطأ: يجب أن يكون الإخراج ملف H5"
    print("✅ الاختبار ناجح! تم حفظ النموذج بنجاح.")
