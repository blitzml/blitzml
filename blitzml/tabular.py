import sys
import numpy as np
import pandas as pd
import importlib.util
from sklearn import preprocessing
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from boruta import BorutaPy
from sklearn.model_selection import cross_validate

class Classification:
    """
    Parameters:
        :param kwargs: is the classifier arguments
    """

    classifiers_map = {
        "RF": RandomForestClassifier,
        "LDA": LinearDiscriminantAnalysis,
        "SVC": SVC,
        "KNN": KNeighborsClassifier,
        "GNB": GaussianNB,
        "LR": LogisticRegression,
        "AB": AdaBoostClassifier,
        "GB": GradientBoostingClassifier,
        "DT": DecisionTreeClassifier,
        "MLP": MLPClassifier
    }

    def __init__(self,
                 train_df,
                 test_df,
                 classifier="RF",
                 class_name = "None",
                 file_path = "None",
                 feature_selection = "None",
                 validation_percentage = 0.1,
                 **kwargs):
        self.train_df = train_df
        self.test_df = test_df
        assert (not (self.train_df.empty) and not (self.test_df.empty))

        if classifier in ['custom','auto']:
            self.classifier = classifier
        else:
            assert (
                classifier in self.classifiers_map.keys()
            ), "Unsupported classifier provided"
            self.classifier = classifier

        self.class_name = class_name
        self.file_path = file_path
        self.kwargs = kwargs
        self.target_col = None
        self.model = None
        self.pred_df = None
        self.metrics_dict = None
        self.target = None
        self.columns_high_corr = None
        self.important_columns =None
        self.used_columns = None
        self.validation_percentage = validation_percentage
        self.validation_df = None

        if feature_selection in ['correlation', 'importance']:
            self.feature_selection = feature_selection
        else:
            self.feature_selection = None



    def get_custom_classifier(self):
        assert(
                self.class_name != "None" and self.file_path != "None"
            ), "Didn't provide the custom classifier arguments!"

        # load module using a class_name and a file_path
        spec = importlib.util.spec_from_file_location(self.class_name, self.file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.class_name] = module
        spec.loader.exec_module(module)

        # returns the class from the loaded module
        return module.__dict__[self.class_name] 

    def preprocess(self):
        train = self.train_df.copy()
        test = self.test_df.copy()
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
        target = None
        for col in train.columns:
            if col not in test.columns:
                target = col
        assert (target != None), 'train_df does not contain target column'
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
        # split for validation
        validation_percentage = self.validation_percentage
        validation_index = int(len(train) * (1-validation_percentage))
        # assign processed dataframes
        self.train_df = train_n.iloc[:validation_index,:].drop(target, axis=1)
        self.validation_df = train_n.iloc[validation_index:,:].drop(target, axis=1)
        self.test_df = test_n
        self.target_col = train_n.iloc[:validation_index,:][target]
        self.true_values = train_n.iloc[validation_index:,:][target]
        self.target = target

    def select_high_correlation(self):
        train_n = self.train_df
        target = self.target
        train_n[target] = self.target_col
        # classify columns by correlation
        corr_df = train_n.corr()
        # drop target raw 
        corr_df.drop(index=target,axis=0,inplace=True)
        # calculate corelation ref
        corr_ref = round(np.percentile(abs(corr_df[target]),33),4)
        columns_high_corr = list(corr_df[(corr_df[target] >= corr_ref)].index) + list(
            corr_df[(corr_df[target] <= -corr_ref)].index
        )
        self.columns_high_corr = columns_high_corr

    def select_important_features(self):
        train = self.train_df
        target_col = self.target_col
        model = RandomForestClassifier(random_state=0)
        # define Boruta feature selection method
        feat_selector = BorutaPy(model, n_estimators='auto', verbose=0, random_state=1)
        # find all relevant features
        feat_selector.fit(train.values, target_col.values)
        cols = list(train.columns)
        importance = list(feat_selector.support_)
        important_columns = []
        for i in range(len(importance)):
            if importance[i]==True:
                important_columns.append(cols[i])
        self.important_columns = important_columns

    def favorable_classifier(self):
        results = pd.DataFrame(columns=["Classifier", "Avg_Accuracy", "Avg_F1_Score"])
        for name, clf in self.classifiers_map.items():
            model = clf
            cv_results = cross_validate(
                model, self.train_df[self.used_columns], self.target_col , cv=10,
                scoring=(['accuracy', 'f1'])
            )

            results = results.append({
                "Classifier": name,
                "Avg_Accuracy": cv_results['test_accuracy'].mean(),
                "Avg_F1_Score": cv_results['test_f1'].mean()
            }, ignore_index=True)
            
        results["Avg_Overall"] = (results["Avg_Accuracy"] + results["Avg_F1_Score"]) / 2
        results = results.sort_values("Avg_Overall", ascending=False)
        return self.classifiers_map[results.iloc[0,:]['Classifier']]

    def train_the_model(self):
        if self.feature_selection == 'correlation':
            self.select_high_correlation()
            self.used_columns = self.columns_high_corr
        elif self.feature_selection == 'importance':
            self.select_important_features()
            self.used_columns = self.important_columns
        elif self.feature_selection == None:
            self.used_columns = list(self.train_df.columns)
        # use high correlation columns only in training
        X = self.train_df[self.used_columns]
        y = self.target_col
        if self.classifier == "custom":
            classifier = self.get_custom_classifier()
        elif self.classifier == 'auto':
            classifier = self.favorable_classifier()
        else:
            classifier = self.classifiers_map[self.classifier]

        self.model = classifier(**self.kwargs)
        self.model.fit(X, y)

    def gen_pred_df(self, df):
        istest=False
        target = self.target
        if df is self.test_df:
            istest=True
        preds = self.model.predict(df[self.used_columns])
        df[target] = preds
        # assign to self.pred_df only if inputs the test df
        if istest:
            self.pred_df = df
        else: # if input is validation
            return df

    def gen_metrics_dict(self):
        # If the user calls this function before gen_pred_df()
        predected = self.gen_pred_df(self.validation_df)

        local_target = self.target
        x = self.true_values
        y = predected[local_target]
        acc = round(accuracy_score(x, y),2)
        f1 = round(f1_score(x, y),2)
        pre = round(precision_score(x, y),2)
        recall = round(recall_score(x, y),2)
        tn, fp, fn, tp = confusion_matrix(x, y).ravel()
        specificity = round(tn / (tn + fp),2)

        dict_metrics = {
            "Accuracy": acc,
            "f1": f1,
            "Precision": pre,
            "Recall": recall,
            "Specificity": specificity,
        }

        # assign the resulting dictionary to self.metrics_dict
        self.metrics_dict = dict_metrics
    
    # run function
    def run(self):
        self.preprocess()
        self.train_the_model()
        self.gen_pred_df(self.test_df)
        self.gen_metrics_dict()


