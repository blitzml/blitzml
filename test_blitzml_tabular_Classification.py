import pandas as pd
from blitzml.tabular import Classification
import pytest

train_df = pd.read_csv("auxiliary/datasets/banknote/train.csv")
test_df = pd.read_csv("auxiliary/datasets/banknote/test.csv")


def test_using_cross_validation():
    auto = Classification(
        train_df,
        test_df,
        classifier='RF',
        cross_validation_k_folds = 5
        )
    auto.run()
    assert len(auto.metrics_dict['cross_validation_score']) == 5
    assert auto.metrics_dict['accuracy'] > 0 

def test_using_train_validation_curve():
    auto = Classification(
        train_df,
        test_df,
        classifier='RF',
        )
    auto.run()
    assert auto.metrics_dict['accuracy'] > 0 
    assert len(auto.accuracy_history()['y1'])>0


def test_validation_percent_greater_than_90_percent_fail():
    with pytest.raises(AssertionError):
        auto = Classification(
            train_df,
            test_df,
            classifier='RF',
            validation_percentage = 0.91
            )
        auto.run()

def test_different_feature_selection_modes():
    modes = ["importance", "correlation", "none"]
    for mode in modes:
        auto = Classification(
            train_df,
            test_df,
            classifier='RF',
            feature_selection = mode
            )
        auto.run()
        assert auto.metrics_dict['accuracy'] > 0    
def test_train_dataset_without_target_column_fails():
    with pytest.raises(AssertionError):
        auto = Classification(
            train_df.drop('class', axis = 1),
            test_df,
            classifier='RF'
            )
        auto.run()
    
def test_classifiers():
    classifier_list = ["RF","LDA","SVC","KNN","GNB","LR","AB","GB","DT","MLP"]
    for classifier in classifier_list:
        auto = Classification(
            train_df,
            test_df,
            classifier=classifier
            )
        auto.run()
        assert auto.metrics_dict['accuracy'] > 0

def test_using_auto_classifier():
    auto = Classification(
        train_df,
        test_df,
        classifier='auto'
        )
    auto.run()
    assert auto.metrics_dict['accuracy'] > 0

def test_using_different_datasets():
    datasets = ['titanic', 'banknote', 'liqure quality']
    for dataset in datasets:
        train_df = pd.read_csv(f"auxiliary/datasets/{dataset}/train.csv")
        test_df = pd.read_csv(f"auxiliary/datasets/{dataset}/test.csv")
        auto = Classification(
            train_df,
            test_df,
            classifier='RF'
            )
        auto.run()
        assert auto.metrics_dict['accuracy'] > 0

def test_using_custom_classifier():
    auto = Classification(
        train_df,
        test_df,
        classifier='custom',
        class_name = "classifier",
        file_path = "auxiliary/scripts/dummy.py",
        )
    auto.run()
    assert auto.metrics_dict['accuracy'] > 0

def test_using_unsupported_average_type_fails():
    train_df = pd.read_csv("auxiliary/datasets/liqure quality/train.csv")
    test_df = pd.read_csv("auxiliary/datasets/liqure quality/test.csv")
    with pytest.raises(ValueError):
        auto = Classification(
            train_df,
            test_df,
            classifier='RF',
            average_type = "cheese"
            )
        auto.run()
def test_using_wrong_custom_classifier_fails():
    with pytest.raises(KeyError):
        auto = Classification(
            train_df,
            test_df,
            classifier='custom',
            class_name = "worng-class-name",
            file_path = "auxiliary/scripts/dummy.py",
            )
        auto.run()
        
def test_using_wrong_custom_classifier_file_path_fails():
    with pytest.raises(FileNotFoundError):
        auto = Classification(
            train_df,
            test_df,
            classifier='custom',
            class_name = "worng-class-name",
            file_path = "auxiliary/scripts/daaa.py",
            )
        auto.run()

def test_using_unsupported_classifier_fails():
    with pytest.raises(AssertionError):
        auto = Classification(
            train_df,
            test_df,
            classifier='Batman'
            )
        auto.run()

def test_using_empty_train_df_fails():
    with pytest.raises(AssertionError):
        train_df = pd.DataFrame()
        auto = Classification(
            train_df,
            test_df,
            classifier='RF'
            )
        auto.run()

def test_using_empty_test_df_fails():
    with pytest.raises(AssertionError):
        test_df = pd.DataFrame()
        auto = Classification(
            train_df,
            test_df,
            classifier='RF'
            )
        auto.run()