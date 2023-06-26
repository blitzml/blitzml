import pandas as pd
from blitzml.tabular import TimeSeries
import pytest

train_df = pd.read_csv("auxiliary/datasets/Alcohol_Sales/train.csv")
test_df =  pd.read_csv("auxiliary/datasets/Alcohol_Sales/test.csv")

def test_different_feature_selection_modes():
    modes = ["importance", "correlation", "none"]
    for mode in modes:
        auto = TimeSeries(
            train_df,
            test_df,
            regressor='RF',
            feature_selection = mode
            )
        auto.run()
        assert auto.metrics_dict['root_mean_squared_error'] < 1000000 

def test_using_train_validation_curve():
    auto = TimeSeries(
        train_df,
        test_df,
        regressor='SVR'
        )
    auto.run()
    assert auto.metrics_dict['root_mean_squared_error'] < 1000000
    assert len(auto.RMSE_history()['y1'])>0

def test_train_dataset_without_target_column_fails():
    with pytest.raises(AssertionError):
        auto = TimeSeries(
            train_df.drop('Sales', axis = 1),
            test_df,
            regressor='SVR'
            )
        auto.run()

def test_regressors():
    regressor_list =["RF","KNN","SVR","DT","GPR","LR","LSS","RDG","GB","MLP"]
    for regressor in regressor_list:
        auto = TimeSeries(
            train_df,
            test_df,
            regressor=regressor
            )
        auto.run()
        assert auto.metrics_dict['root_mean_squared_error'] < 100000000

def test_using_different_datasets():
    datasets = ['HospitalityEmployees', 'EnergyIndex','store sales']
    for dataset in datasets:
        train_df = pd.read_csv(f"auxiliary/datasets/{dataset}/train.csv")
        test_df = pd.read_csv(f"auxiliary/datasets/{dataset}/test.csv")
        auto = TimeSeries(
            train_df,
            test_df,
            regressor='RF'
            )
        auto.run()
        assert auto.metrics_dict['root_mean_squared_error'] < 1000000 

def test_using_auto_regressor():
    auto = TimeSeries(
        train_df,
        test_df,
        regressor='RF',
        feature_selection = "importance",
        )
    auto.run()
    assert auto.metrics_dict['root_mean_squared_error'] < 100000000 

def test_using_custom_regressor():
    auto = TimeSeries(
        train_df,
        test_df,
        regressor='custom',
        class_name = "classifier",
        file_path = "auxiliary/scripts/dummy.py",
        )
    auto.run()
    assert auto.metrics_dict['root_mean_squared_error'] < 1000000

def test_using_wrong_custom_regressor_fails():
    with pytest.raises(KeyError):
        auto = TimeSeries(
            train_df,
            test_df,
            regressor='custom',
            class_name = "worng-class-name",
            file_path = "auxiliary/scripts/dummy.py",
            )
        auto.run()

def test_using_wrong_custom_classifier_file_path_fails():
    with pytest.raises(FileNotFoundError):
        auto = TimeSeries(
            train_df,
            test_df,
            regressor='custom',
            class_name = "worng-class-name",
            file_path = "auxiliary/scripts/daaa.py",
            )
        auto.run()

def test_using_unsupported_regressor_fails():
    with pytest.raises(AssertionError):
        auto = TimeSeries(
            train_df,
            test_df,
            regressor='Batman'
            )
        auto.run()

def test_using_empty_train_df_fails():
    with pytest.raises(AssertionError):
        train_df = pd.DataFrame()
        auto = TimeSeries(
            train_df,
            test_df,
            regressor='SVR',
            )
        auto.run()

def test_using_empty_test_df_fails():
    with pytest.raises(AssertionError):
        test_df = pd.DataFrame()
        auto = TimeSeries(
            train_df,
            test_df,
            regressor='SVR'
            )
        auto.run()  

 