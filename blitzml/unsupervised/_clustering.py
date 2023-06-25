import sys
import numpy as np
import pandas as pd
import importlib.util
from sklearn import preprocessing
from sklearn.decomposition import PCA
# ignore pandas warnings 
import warnings
warnings.filterwarnings('ignore')
# metrics imports 
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score,
    davies_bouldin_score
)

# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import mutual_info_regression

from sklearn.ensemble import RandomForestClassifier # for feature importance
from sklearn.cluster import (
    KMeans,
    AffinityPropagation,
    AgglomerativeClustering,
    MeanShift,
    SpectralClustering,
    Birch,
    BisectingKMeans,
    OPTICS,
    DBSCAN
)

class Clustering:
    """
    Parameters:
        :param kwargs: is the clustering_algorithm arguments
    """

    clustering_map = {
    # https://scikit-learn.org/stable/modules/clustering.html
        "KM": KMeans,
        "AP":AffinityPropagation,
        "AC":AgglomerativeClustering,
        "MS":MeanShift,
        "SC":SpectralClustering,
        "Birch":Birch,
        "BKM":BisectingKMeans,
        "OPTICS":OPTICS,
        "DBSCAN":DBSCAN
    }

    def __init__(self,
                 train_df,
                 clustering_algorithm="KM",
                 class_name = "None",
                 file_path = "None",
                 feature_selection = "none",
                 validation_percentage = 0.1,
                 n_clusters = 8,
                 **kwargs):
        self.train_df = train_df
        assert (not (self.train_df.empty))

        if clustering_algorithm in ['custom','auto']:
            self.clustering_algorithm = clustering_algorithm
        else:
            assert (
                clustering_algorithm in self.clustering_map.keys()
            ), "Unsupported clustering_algorithm provided"
            self.clustering_algorithm = clustering_algorithm

        self.class_name = class_name
        self.file_path = file_path
        self.kwargs = kwargs
        self.model = None
        self.pred_df = None
        self.metrics_dict = None
        self.important_columns =None
        self.used_columns = None
        self.true_values = None
        self.validation_percentage = validation_percentage
        assert (self.validation_percentage<=0.9), "Validation % must be <=0.9"
        self.validation_df = None
        self.scaler = None
        self.n_clusters = n_clusters
        self.algorithms_needing_n_clusters = ['KM', 'AC', 'SC', 'Birch', 'BKM', '']
        if feature_selection in ['correlation', 'importance']:
            self.feature_selection = feature_selection
        else:
            self.feature_selection = None


    def get_custom_clustering_algorithm(self):
        assert(
                self.class_name != "None" and self.file_path != "None"
            ), "Didn't provide the custom clustering_algorithm arguments!"

        # load module using a class_name and a file_path
        spec = importlib.util.spec_from_file_location(self.class_name, self.file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.class_name] = module
        spec.loader.exec_module(module)

        # returns the class from the loaded module
        return module.__dict__[self.class_name] 

    def preprocess(self):
        train = self.train_df.copy()
        # drop duplicates
        train.drop_duplicates(inplace=True)
        # drop rows if contains null data in column have greater than 95% valid data (from train only)
        null_df = train.isnull().mean().to_frame()
        null_df["column"] = null_df.index
        null_df.index = np.arange(null_df.shape[0])
        null_cols = list(null_df[null_df[0].between(0.001, 0.05)]["column"])
        for colmn in null_cols:
            null_index = list(train[train[colmn].isnull()].index)
            train.drop(index=null_index, axis=0, inplace=True)

        df = train
        # drop columns na >= 25%
        null_df = df.isnull().mean().to_frame()
        null_df["column"] = null_df.index
        null_df.index = np.arange(null_df.shape[0])
        null_cols = list(null_df[null_df[0] >= 0.25]["column"])
        df.drop(columns=null_cols, inplace=True)
        # now we should know what is numerical columns and categorical columns
        dtype = df.dtypes.to_frame()
        dtype["column"] = dtype.index
        dtype.index = np.arange(dtype.shape[0])
        cat_colmns = []
        num_colmns = []
        num_values = [
            "float64",
            "int64",
            "uint8",
            "int32",
            "int8",
            "int16",
            "uint16",
            "uint32",
            "uint64",
            "float_",
            "float16",
            "float32",
            "int_",
            "int",
            "float",
        ]
        for i in range(len(dtype.column)):
            if "object" in str(dtype.iloc[i, 0]):
                cat_colmns.append(dtype.column[i])
            elif str(dtype.iloc[i, 0]) in num_values:
                num_colmns.append(dtype.column[i])

        columns_genres = [cat_colmns, num_colmns]
        # drop columns has more than 7 unique values (from categorical columns)
        columns_has_2 = []
        columns_has_3to7 = []
        columns_to_drop = []
        for c_col in cat_colmns:
            if df[c_col].nunique() > 7:
                columns_to_drop.append(c_col)
            elif 3 <= df[c_col].nunique() <= 7:
                columns_has_3to7.append(c_col)
            else:
                columns_has_2.append(c_col)
        # fillna in categorical columns
        df[cat_colmns] = df[cat_colmns].fillna("unknown")
        # fillna in numerical columns
        for column in num_colmns:
            df[column].fillna(value=df[column].mean(), inplace=True)
        # scale numerical columns
        # scaler = preprocessing.StandardScaler()
        # for num_col in num_colmns:
        #     df[num_col] = scaler.fit_transform(np.array(df[num_col]).reshape(1,-1)).reshape(-1,1)
        # now we can drop without raising error
        df.drop(columns = columns_to_drop, inplace=True)
        # encode the categorical
        encoder = preprocessing.LabelEncoder()
        for col in columns_has_2:
            df[col] = encoder.fit_transform(
                df[col]
            )  # encode columns has 2 unique values in the same column 0, 1
        # encode columns has 3-7 unique values
        for cat in columns_has_3to7:
            df = pd.concat([df, pd.get_dummies(df[cat], prefix=cat)], axis=1)
            df = df.drop([cat], axis=1)
        
        # assign processed dataframes
        self.train_df = df


    def select_important_features(self):
        # https://towardsdatascience.com/interpretable-k-means-clusters-feature-importances-7e516eeb8d3c
        # Unsupervised to supervised
        # For each cluster create one-vs-all classification
        # first, get clusters labels
        model = KMeans(n_clusters=int(len(self.train_df.columns)/2))
        model.fit(self.train_df)
        labels = model.predict(self.train_df)
        # create one_vs_all matrix 
        # each row represents a cluster with one-vs-all encoding
        one_vs_all = []
        for i in range(np.max(labels) + 1):
            one_vs_all.append([])
            for label in labels:
                if label == i:
                    one_vs_all[i].append(1)
                else:
                    one_vs_all[i].append(0)
        # fitting a random forest classifier on each row of the matrix
        # summing all feature_importances_ for every classifier
        importance_sum = np.empty(len(self.train_df.columns))
        for cluster_labels in one_vs_all:
            clf = RandomForestClassifier(random_state=1)
            clf.fit(self.train_df, cluster_labels)
            importance_sum+=clf.feature_importances_
        # Reverse sort
        sorted_feature_weight_idxes = np.argsort(importance_sum)[::-1] 
        # Get the most important features names and weights
        most_important_features = np.take_along_axis(
            np.array(self.train_df.columns.tolist()), 
            sorted_feature_weight_idxes, axis=0)

        most_important_weights = np.take_along_axis(
            np.array(importance_sum), 
            sorted_feature_weight_idxes, axis=0)

        # choose only features that contribute to 90% of the weight_sum
        weights_sum = sum(most_important_weights)
        temp_sum = 0
        threshold_index = 0
        for i, weight in enumerate(most_important_weights):
            temp_sum += weight
            if temp_sum >= (0.9*weights_sum):
                threshold_index = i
                break
        # return only features that contribute 90% of the importance
        self.important_columns = most_important_features[:threshold_index]

    def used_cols(self):
        if self.feature_selection == 'importance':
            self.select_important_features()
            self.used_columns = self.important_columns
        elif self.feature_selection == None:
            self.used_columns = list(self.train_df.columns)

    def auto_clustering(self):
        # Auto number of clusters
        # https://towardsdatascience.com/how-many-clusters-6b3f220f0ef5
        # Find the best performing algorithm
        results = pd.DataFrame(columns=["clustering_algorithm", "silhouette_score"])
        for name, algorithm in self.clustering_map.items():
            model = algorithm()
            preds = model.fit_predict(self.train_df[self.used_columns])
            # skip models that predicted only one cluster
            if len(np.unique(preds)) == 1:
                continue
            sil_score = silhouette_score(self.train_df[self.used_columns], preds)
            results = results.append({
                "clustering_algorithm": name,
                "silhouette_score": sil_score,
            }, ignore_index=True)
            
        best_clutering_algorithm = results.sort_values("silhouette_score", ascending=False).iloc[0,:]['clustering_algorithm']
        # find the number of clusters
        if best_clutering_algorithm in self.algorithms_needing_n_clusters:
            self.n_clusters = self.get_n_clusters()    
    
        return self.clustering_map[best_clutering_algorithm]
    
    def get_n_clusters(self):
        best_n = 0
        best_sil_score = -1
        for i in range(2, 25):
            model = KMeans(n_clusters = i)
            preds = model.fit_predict(self.train_df[self.used_columns])
            sil_score = silhouette_score(self.train_df[self.used_columns], preds)
            if sil_score > best_sil_score:
                best_sil_score = sil_score
                best_n = i
        return best_n

    def train_the_model(self):
        self.used_cols()
        # use selected columns only in training
        X = self.train_df[self.used_columns]
        if self.clustering_algorithm == "custom":
            clustering_algorithm = self.get_custom_clustering_algorithm()
        elif self.clustering_algorithm == 'auto':
            clustering_algorithm = self.auto_clustering()
        else:
            clustering_algorithm = self.clustering_map[self.clustering_algorithm]
        # for algorithm accepting n_clusters provide the class parameter self.n_clustering
        # note: when using auto_clustering, best self.n_clustering value is calculated 
        try:
            self.model = clustering_algorithm(self.n_clusters, **self.kwargs)
        except:
            self.model = clustering_algorithm(**self.kwargs)
        self.model.fit(X)

    def clustering_visualization(self):
        # https://www.datacamp.com/tutorial/principal-component-analysis-in-python
        # crisp vs overlapping clusters
        # some information is lost in dimensionality reduction

        # first, scale the features
        scaler = preprocessing.StandardScaler()
        X_scaled = scaler.fit_transform(self.train_df[self.used_columns])
        # reduce the dimension to 2-d using PCA
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        components_df = pd.DataFrame(data = components
             , columns = ['principal_component_1', 'principal_component_2'])

        # predict the cluster labels
        cluster_labels = self.model.fit_predict(self.train_df[self.used_columns])

        title = str(self.model).split('(')[0] +  ' clusters'

        data = {
            'principal_component_1':list(components_df['principal_component_1']),
            'principal_component_2':list(components_df['principal_component_2']),
            'cluster_labels':cluster_labels,
            'title':title,
        }
        return data

    def gen_pred_df(self):
        try:
            preds = self.model.predict(self.train_df[self.used_columns])
        except: # for OPTICS and DBSCAN clustering algorithm
            preds = self.model.fit_predict(self.train_df[self.used_columns])
        df = self.train_df
        df['cluster'] = preds
        self.pred_df = df

    def gen_metrics_dict(self):
        # https://towardsdatascience.com/7-evaluation-metrics-for-clustering-algorithms-bdc537ff54d2
        try:
            preds = self.model.predict(self.train_df[self.used_columns])
        except: # for OPTICS and DBSCAN clustering algorithm
            preds = self.model.fit_predict(self.train_df[self.used_columns])
        if len(np.unique(preds)) < 2:
            sil_score = 0 
            cal_har_score = 0
            dav_boul_score = 0
        else:
            sil_score = silhouette_score(self.train_df[self.used_columns], preds)
            cal_har_score = calinski_harabasz_score(self.train_df[self.used_columns], preds)
            dav_boul_score = davies_bouldin_score(self.train_df[self.used_columns], preds)

        dict_metrics = {
            "silhouette_score": sil_score,
            "calinski_harabasz_score": cal_har_score,
            "davies_bouldin_score": dav_boul_score,
            "n_clusters":len(np.unique(preds))
        }
        self.metrics_dict = dict_metrics
    
    def run(self):
        self.preprocess()
        self.train_the_model()
        self.gen_pred_df()
        self.gen_metrics_dict()