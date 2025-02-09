import mlflow
import mlflow.tensorflow
from transformers import TrainerCallback
from tensorflow.keras.callbacks import Callback


class MLFlow:
    def __init__(self, experiment_name: str):
        """تهيئة MLflow مع تخزين محلي."""
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(
            "file:///E:/Task7/mlruns"
        )  # تأكد من ضبط المسار بشكل صحيح
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str):
        """بدء جلسة جديدة مع إنهاء الجلسة السابقة إن وجدت."""
        if mlflow.active_run():
            mlflow.end_run()
        self.run = mlflow.start_run(run_name=run_name)
        print(f"✅ بدأ تشغيل MLflow مع اسم الجلسة: {run_name}")

    def log_params(self, params: dict):
        """تسجيل معلمات التدريب."""
        mlflow.log_params(params)
        print(f"✅ تسجيل المعلمات: {params}")

    def log_metrics(self, metrics: dict):
        """تسجيل المقاييس أثناء التدريب."""
        mlflow.log_metrics(metrics)
        print(f"✅ تسجيل المقاييس: {metrics}")

    def log_model(self, model, model_path: str):
        """حفظ النموذج في MLflow."""
        mlflow.tensorflow.log_model(model, artifact_path=model_path)
        print(f"✅ تم تسجيل النموذج في MLflow: {model_path}")

    def end_run(self):
        """إنهاء التجربة."""
        mlflow.end_run()
        print("🏁 تجربة MLflow انتهت.")


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
