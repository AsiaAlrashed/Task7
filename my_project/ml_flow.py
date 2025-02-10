import mlflow
import mlflow.tensorflow
from transformers import TrainerCallback
from tensorflow.keras.callbacks import Callback


class MLFlow:
    def __init__(self, experiment_name: str):
        """
        Configure MLflow with local storage
        """
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(
            "file:///E:/Task7/mlruns"
        )  
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        """
        Start a new session and end the previous session if any.
        """
        if mlflow.active_run():
            mlflow.end_run()
        self.run = mlflow.start_run(run_name=run_name)
        print(f"Start MLflow with session name: {run_name}")

    def log_params(self, params: dict):
        """
        Register training parameters
        """
        mlflow.log_params(params)
        print(f"Register parameters: {params}")

    def log_metrics(self, metrics: dict):
        """
        Record metrics during training.
        """
        mlflow.log_metrics(metrics)
        print(f"Recording metrics: {metrics}")

    def log_model(self, model, model_path: str):
        """
        Save the model in MLflow.
        """
        mlflow.tensorflow.log_model(model, artifact_path=model_path)
        print(f"The model is registered in MLflow: {model_path}")

    def end_run(self):
        """
        End the experiment.
        """
        mlflow.end_run()
        print("MLflow experiment is over")


class MLflowCallback(Callback):
    def __init__(self, mlflow_tracker):
        super().__init__()
        self.mlflow_tracker = mlflow_tracker

    def on_train_begin(self, logs=None):
        self.mlflow_tracker.start_run(run_name="LSTM Text Classification")
        self.mlflow_tracker.log_params(self.params)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.mlflow_tracker.log_metrics(logs)

    def on_train_end(self, logs=None):
        self.mlflow_tracker.log_model(self.model, "lstm_text_classifier")
        self.mlflow_tracker.end_run()
