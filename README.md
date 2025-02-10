In this project, I developed a text classification model using LSTM network to identify sentiment in tweets. The project includes the following steps:

Data extraction: I used Pandas library to load training and test data from CSV files.

Data processing: I created a TextProcessor class to train the Tokenizer and encode the labels.

Model training: I built an LSTM model using TensorFlow library and trained it on the processed data.

API development: I created an API using FastAPI to receive new texts and return predictions.


And in this project, I used Mage, an advanced data pipeline development and management platform, to facilitate the process of building a pipeline for processing and analyzing text data. Mage provided an interactive environment that supported multiple programming languages ​​such as Python, SQL, and R, allowing me to efficiently integrate and transform data.

I started by loading training and test data using Pandas, then converted the texts to numeric sequences using TextProcessor. Next, I trained an LSTM model to classify texts, while tracking the training process and recording the results using MLflow.

Using Mage, I was able to design an integrated data pipeline that includes data extraction, processing, model training, and evaluation, while ensuring ease of monitoring and management. Mage provides powerful tools for developing data pipelines, making it easy to work with data from multiple sources and perform complex transformations with high efficiency.

Additionally, I created a custom class for MLflow to track the training process and record models, parameters, and metrics. This integration with MLflow allowed me to monitor and record the model performance in a structured manner, facilitating future analysis and improvement.

In short, Mage simplifies and speeds up the data pipeline development process, while providing advanced tools for tracking and improving machine learning models.


dataset: Task7/my_project/data/
Data extraction: Task7/my_project/data_loaders/
Data processing: Task7/my_project/transformers/transform.py
Model training: Task7/my_project/transformers/lstm_model.py
API development : Task7/my_project/custom/main.py
