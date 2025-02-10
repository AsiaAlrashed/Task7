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
    Training an LSTM model on text data.
    
    Args:
    - data (tuple): Consists of `df_train` and `df_test`, each containing:
    * `processed_text`: A list of lists containing the encoded texts.
    * `encoded_label`: The labels converted to numbers.
    
    Returns:
    - str: Path to save the trained model.
    """

    df_train, df_test = data  

    # Convert data to NumPy arrays
    X_train = np.array(df_train["processed_text"].tolist())
    y_train = np.array(df_train["encoded_label"].tolist())

    X_test = np.array(df_test["processed_text"].tolist())
    y_test = np.array(df_test["encoded_label"].tolist())

    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Setting up model variables
    vocab_size = 5000  # Same size as used in `TextProcessor`
    embedding_dim = 64
    max_length = X_train.shape[1]

    # Create an LSTM model
    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            LSTM(64, return_sequences=True),
            Dropout(0.5),
            LSTM(32),
            Dropout(0.5),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),  
        ]
    )

    # Assemble the model
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    #MLflow setup
    mlflow_tracker = MLFlow(experiment_name="LSTM_Text_Classification")
    mlflow_callback = MLflowCallback(mlflow_tracker)

    # Model training
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[mlflow_callback],  # Add MLflow as Callback
    )

    # Evaluate the model on test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model accuracy on test data: {accuracy:.4f}")

    # save model
    model_path = "E:\\Task7\\lstm_text_classifier.h5"
    model.save(model_path)
    print(f"The form has been saved in: {model_path}")

    # Register the form in MLflow
    mlflow_tracker.log_model(model, "lstm_text_classifier")
    mlflow_tracker.end_run()

    return model_path


@test
def test_output(output, *args) -> None:
    """
    Verify the output validity.
    - Ensure that the path of the resulting model is correct.
    """
    assert isinstance(
        output, str
    ), "Error: Output must be text containing the form path"
    assert output.endswith(".h5"), "Error: Output must be H5 file"
    print("Test successful! Form saved successfully.")
