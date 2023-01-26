import pandas as pd
from blitzml.tabular import Classification
import pytest

train_df = pd.read_csv("auxiliary/datasets/banknote/train.csv")
test_df = pd.read_csv("auxiliary/datasets/banknote/test.csv")
ground_truth_df = pd.read_csv("auxiliary/datasets/banknote/ground_truth.csv")

def test_classifiers():
    classifier_list = ["RF","LDA","SVC","KNN","GNB","LR","AB","GB","DT","MLP"]
    for classifier in classifier_list:
        auto = Classification(
            train_df,
            test_df,
            ground_truth_df,
            classifier=classifier
            )
        auto.run()

def test_using_different_datasets():
    datasets = ['titanic', 'banknote']
    for dataset in datasets:
        train_df = pd.read_csv(f"auxiliary/datasets/{dataset}/train.csv")
        test_df = pd.read_csv(f"auxiliary/datasets/{dataset}/test.csv")
        ground_truth_df = pd.read_csv(f"auxiliary/datasets/{dataset}/ground_truth.csv")
        auto = Classification(
            train_df,
            test_df,
            ground_truth_df,
            classifier='RF'
            )
        auto.run()

def test_using_custom_classifier():
    auto = Classification(
        train_df,
        test_df,
        ground_truth_df,
        classifier='custom',
        class_name = "classifier",
        file_path = "auxiliary/scripts/dummy.py",
        )
    auto.run()

def test_using_wrong_custom_classifier_fails():
    with pytest.raises(KeyError):
        auto = Classification(
            train_df,
            test_df,
            ground_truth_df,
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
            ground_truth_df,
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
            ground_truth_df,
            classifier='Batman'
            )
        auto.run()

def test_using_empty_train_df_fails():
    with pytest.raises(AssertionError):
        train_df = pd.DataFrame()
        auto = Classification(
            train_df,
            test_df,
            ground_truth_df,
            classifier='RF'
            )
        auto.run()
def test_using_empty_test_df_fails():
    with pytest.raises(AssertionError):
        test_df = pd.DataFrame()
        auto = Classification(
            train_df,
            test_df,
            ground_truth_df,
            classifier='RF'
            )
        auto.run()

def test_using_empty_ground_truth_df_fails():
    with pytest.raises(AssertionError):
        ground_truth_df = pd.DataFrame()
        auto = Classification(
            train_df,
            test_df,
            ground_truth_df,
            classifier='RF'
            )
        auto.run()