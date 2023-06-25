import pandas as pd
from blitzml.unsupervised import Clustering
import pytest

train_df = pd.read_csv("auxiliary/datasets/customer personality/train.csv")


def test_using_clustering_visualization():
    auto = Clustering(
        train_df,
        clustering_algorithm="KM"
        )
    auto.run()
    vis_dict = auto.clustering_visualization()
    assert len(vis_dict['principal_component_1']) > 0


def test_different_feature_selection_modes():
    modes = ["importance", "none"]
    for mode in modes:
        auto = Clustering(
            train_df,
            clustering_algorithm="KM",
            feature_selection = mode
            )
        auto.run()
        assert auto.metrics_dict['silhouette_score'] >= -1  

def test_clustering_algorithms():
    algoritms_list = ["KM", "AP", "AC", "MS", "SC", "Birch", "BKM", "OPTICS", "DBSCAN"]
    for algo in algoritms_list:
        auto = Clustering(
            train_df,
            clustering_algorithm=algo
            )
        auto.run()
        assert auto.metrics_dict['silhouette_score'] >= -1  

def test_using_auto_clustering():
    auto = Clustering(
        train_df,
        clustering_algorithm='auto'
        )
    auto.run()
    assert auto.metrics_dict['silhouette_score'] >= -1  


def test_using_different_datasets():
    datasets = ['customer personality', 'coffee quality', 'dry bean']
    for dataset in datasets:
        train_df = pd.read_csv(f"auxiliary/datasets/{dataset}/train.csv")
        auto = Clustering(
            train_df,
            clustering_algorithm='KM'
            )
        auto.run()
        assert auto.metrics_dict['silhouette_score'] >= -1


def test_using_custom_clustering_algorithm():
    auto = Clustering(
        train_df,
        clustering_algorithm='custom',
        class_name = "clustering",
        file_path = "auxiliary/scripts/dummy.py",
        )
    auto.run()
    assert auto.metrics_dict['silhouette_score'] >= -1


def test_using_wrong_custom_clustering_fails():
    with pytest.raises(KeyError):
        auto = Clustering(
            train_df,
            clustering_algorithm='custom',
            class_name = "wrong-clustering",
            file_path = "auxiliary/scripts/dummy.py",
            )
        auto.run()


def test_using_wrong_custom_clustering_file_path_fails():
    with pytest.raises(FileNotFoundError):
        auto = Clustering(
            train_df,
            clustering_algorithm='custom',
            class_name = "clustering",
            file_path = "auxiliary/scripts/dummmmy.py",
            )
        auto.run()


def test_using_unsupported_clustering_algorithm_fails():
    with pytest.raises(AssertionError):
        auto = Clustering(
            train_df,
            clustering_algorithm='Batman'
            )
        auto.run()

def test_using_empty_train_df_fails():
    with pytest.raises(AssertionError):
        train_df = pd.DataFrame()
        auto = Clustering(
            train_df,
            clustering_algorithm='KM'
            )
        auto.run()

def test_generating_pred_df():
    auto = Clustering(
        train_df,
        clustering_algorithm='KM',
        )
    auto.run()
    assert len(auto.pred_df) > 0
