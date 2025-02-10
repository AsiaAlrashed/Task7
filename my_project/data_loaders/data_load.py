import os
import pandas as pd

if "data_loader" not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if "test" not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def extract_data(*args, **kwargs):
    """
    Load training and test data using Pandas.
    """
    train_path = "E:\\Task7\\my_project\\data\\Sentimentdataset_train.csv"
    test_path = "E:\\Task7\\my_project\\data\\Sentimentdataset_test.csv"

    # Make sure the files are there
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("ملف التدريب أو الاختبار غير موجود")

    # Load data
    df_train = pd.read_csv(train_path)[['tweet_id', 'new_tweet_content', 'Label']]
    df_test = pd.read_csv(test_path)[['tweet_id', 'new_tweet_content', 'Label']]

    return df_train, df_test

@test
def test_output(output, *args) -> None:
    """
    Verify data after uploading.
    """
    assert isinstance(output, tuple), "Output should be a tuple (train_df, test_df)"
    train_df, test_df = output

    assert train_df is not None and not train_df.empty, "Train dataset is empty"
    assert test_df is not None and not test_df.empty, "Test dataset is empty"
