import pandas as pd
from blitzml.tabular import Regression
import pytest

train_df = pd.read_csv("auxiliary/datasets/house prices/train.csv")
test_df = pd.read_csv("auxiliary/datasets/house prices/test.csv")


def test_using_cross_validation():
    auto = Regression(
        train_df,
        test_df,
        algorithm='SVR',
        cross_validation_k_folds = 5
        )
    auto.run()
    assert len(auto.metrics_dict['cross_validation_score']) == 5
    assert auto.metrics_dict['root_mean_squared_error'] < 1000000

def test_using_train_validation_curve():
    auto = Regression(
        train_df,
        test_df,
        algorithm='SVR'
        )
    auto.run()
    assert auto.metrics_dict['root_mean_squared_error'] < 1000000
    assert len(auto.rmse_history()['y1'])>0

def test_validation_percent_greater_than_90_percent_fail():
    with pytest.raises(AssertionError):
        auto = Regression(
            train_df,
            test_df,
            algorithm='SVR',
            validation_percentage = 0.91
            )
        auto.run()

def test_different_feature_selection_modes():
    modes = ["importance", "correlation", "none"]
    for mode in modes:
        auto = Regression(
            train_df,
            test_df,
            algorithm='SVR',
            feature_selection = mode
            )
        auto.run()
        assert auto.metrics_dict['root_mean_squared_error'] < 1000000  

def test_train_dataset_without_target_column_fails():
    with pytest.raises(AssertionError):
        auto = Regression(
            train_df.drop('SalePrice', axis = 1),
            test_df,
            algorithm='SVR'
            )
        auto.run()
def test_regressors():
    regressor_list = ["RF","KNN","SVR","DT","GPR","LR","LSS","RDG","GB","MLP"]
    for regressor in regressor_list:
        auto = Regression(
            train_df,
            test_df,
            algorithm=regressor
            )
        auto.run()
        assert auto.metrics_dict['root_mean_squared_error'] < 100000000 

def test_using_auto_regressor():
    auto = Regression(
        train_df,
        test_df,
        algorithm='auto'
        )
    auto.run()
    assert auto.metrics_dict['root_mean_squared_error'] < 100000000 

def test_using_different_datasets():
    datasets = ['meat consumption', 'house prices']
    for dataset in datasets:
        train_df = pd.read_csv(f"auxiliary/datasets/{dataset}/train.csv")
        test_df = pd.read_csv(f"auxiliary/datasets/{dataset}/test.csv")
        auto = Regression(
            train_df,
            test_df,
            algorithm='SVR'
            )
        auto.run()
        assert auto.metrics_dict['root_mean_squared_error'] < 1000000

def test_using_custom_regressor():
    auto = Regression(
        train_df,
        test_df,
        algorithm='custom',
        class_name = "classifier",
        file_path = "auxiliary/scripts/dummy.py",
        )
    auto.run()
    assert auto.metrics_dict['root_mean_squared_error'] < 1000000

def test_using_wrong_custom_regressor_fails():
    with pytest.raises(KeyError):
        auto = Regression(
            train_df,
            test_df,
            algorithm='custom',
            class_name = "worng-class-name",
            file_path = "auxiliary/scripts/dummy.py",
            )
        auto.run()

def test_using_wrong_custom_classifier_file_path_fails():
    with pytest.raises(FileNotFoundError):
        auto = Regression(
            train_df,
            test_df,
            algorithm='custom',
            class_name = "worng-class-name",
            file_path = "auxiliary/scripts/daaa.py",
            )
        auto.run()

def test_using_unsupported_regressor_fails():
    with pytest.raises(AssertionError):
        auto = Regression(
            train_df,
            test_df,
            algorithm='Batman'
            )
        auto.run()

def test_using_empty_train_df_fails():
    with pytest.raises(AssertionError):
        train_df = pd.DataFrame()
        auto = Regression(
            train_df,
            test_df,
            algorithm='SVR'
            )
        auto.run()

def test_using_empty_test_df_fails():
    with pytest.raises(AssertionError):
        test_df = pd.DataFrame()
        auto = Regression(
            train_df,
            test_df,
            algorithm='SVR'
            )
        auto.run()