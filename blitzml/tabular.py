import joblib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC


class Classification:
    classifiers_map = {
        "RF": RandomForestClassifier,
        "LDA": LinearDiscriminantAnalysis,
        "SVC": SVC,
    }

    def __init__(
        self, train_df, test_df, ground_truth_df, classifier="RF", n_estimators=100
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.ground_truth_df = ground_truth_df

        assert (
            classifier in self.classifiers_map.keys()
        ), "Unsupported classifier provided"
        self.classifier = classifier

        self.n_estimators = n_estimators
        self.target_col = None
        self.model = None
        self.pred_df = None
        self.metrics_dict = None
        self.target = None
        self.columns_high_corr = None

    def preprocess(self):
        train = self.train_df
        test = self.test_df
        # drop duplicates
        train.drop_duplicates(inplace=True)
        # drop raws if contains null data in column have greater than 95% valid data (from train only)
        null_df = train.isnull().mean().to_frame()
        null_df["column"] = null_df.index
        null_df.index = np.arange(null_df.shape[0])
        null_cols = list(null_df[null_df[0].between(0.001, 0.05)]["column"])
        for colmn in null_cols:
            null_index = list(train[train[colmn].isnull()].index)
            train.drop(index=null_index, axis=0, inplace=True)
        # get target data (column name,dtype,save values in list)
        for col in train.columns:
            if col not in test.columns:
                target = col
        # get dtype
        dtype = train.dtypes.to_frame()
        dtype["column"] = dtype.index
        dtype.index = np.arange(dtype.shape[0])
        dt = str(dtype[dtype["column"] == target].iloc[0, 0])  # target dtype
        if "int" in dt:
            dt = int
        elif "float" in dt:
            dt = float
        elif "object" in dt:
            dt = str
        else:
            dt = "unknown"
        # save target list
        target_list = train[target]
        # concatinate datasets, first columns must be identical
        train.drop(columns=[target], inplace=True)
        train[target] = target_list
        train[target] = train[target].astype(str)
        test[target] = np.repeat("test_dataset", test.shape[0])
        df = pd.concat([train, test])  # concatinate datasets
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
        columns_genres = [cat_colmns, num_colmns]
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
        # remove target column from lists (not from dataframe)
        columns_genres = [cat_colmns, num_colmns]
        for genre in columns_genres:
            if target in genre:
                genre.remove(target)
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
        # now we can drop without raising error
        df.drop(columns=columns_to_drop, inplace=True)
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
        # split train and test
        train_n = df[df[target] != "test_dataset"]
        test_n = df[df[target] == "test_dataset"].drop(target, axis=1)
        try:
            train_n[target] = train_n[target].astype(dt)
        except:
            pass
        # classify columns by correlation
        corr_df = train_n.corr()
        columns_high_corr = list(corr_df[(corr_df[target] >= 0.15)].index) + list(corr_df[(corr_df[target] <= -0.15)].index)
        columns_high_corr.remove(target)
        # assign processed dataframes
        self.train_df = train_n.drop(target, axis=1)
        self.test_df = test_n
        self.target_col = train_n[target]
        self.target = target
        self.columns_high_corr = columns_high_corr

    def train_the_model(self):
        columns_high_corr = self.columns_high_corr
        # use high correlation columns only in training
        X = self.train_df[columns_high_corr]
        y = self.target_col

        classifier = self.classifiers_map[self.classifier]

        if self.classifier == "RF":
            kwargs = {"n_estimators": self.n_estimators}
        else:
            kwargs = {}

        self.model = classifier(**kwargs)
        self.model.fit(X, y)

    def gen_pred_df(self):
        target = self.target
        columns_high_corr = self.columns_high_corr
        preds = self.model.predict(self.test_df[columns_high_corr])
        # columns should be in submission
        ground_truth_columns = list(self.ground_truth_df.columns)
        ground_truth_columns.remove(target)
        # predict submission
        pred_df = self.test_df[ground_truth_columns]
        pred_df[target] = preds
        # assign to self.pred_df
        self.pred_df = pred_df

    def gen_metrics_dict(self):
        # If the user calls this function before gen_pred_df()
        if self.pred_df.empty:
            self.gen_pred_df()

        localtarget = self.target
        x = self.ground_truth_df[localtarget]
        y = self.pred_df[localtarget]
        acc = accuracy_score(x, y)
        f1 = f1_score(x, y)
        pre = precision_score(x, y)
        recall = recall_score(x, y)
        tn, fp, fn, tp = confusion_matrix(x, y).ravel()
        specificity = tn / (tn + fp)

        dict_metrics = {
            "Accuracy": acc,
            "f1": f1,
            "Precision": pre,
            "Recall": recall,
            "Specificity": specificity,
        }

        # assign the resulting dictionary to self.metrics_dict
        self.metrics_dict = dict_metrics
