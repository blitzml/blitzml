from ._supervised import Supervised_ML
import numpy as np
import matplotlib.pyplot as plt
# ignore pandas warnings 
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
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
from sklearn.model_selection import cross_validate, learning_curve, cross_val_score

class Classification(Supervised_ML):
    """
    Parameters:
        :param kwargs: is the classifier arguments
    """

    algorithms_map = {
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
                 algorithm="RF",
                 class_name = "None",
                 file_path = "None",
                 feature_selection = "none",
                 validation_percentage = 0.1,
                 average_type = 'macro', # for multiclass classification metrics
                 cross_validation_k_folds = 1,
                 **kwargs):
        self.train_df = train_df
        self.test_df = test_df
        assert (not (self.train_df.empty) and not (self.test_df.empty))

        if algorithm in ['custom','auto']:
            self.algorithm = algorithm
        else:
            assert (
                algorithm in self.algorithms_map.keys()
            ), "Unsupported algorithm provided"
            self.algorithm = algorithm

        self.class_name = class_name
        self.file_path = file_path
        self.kwargs = kwargs
        self.target_col = None
        self.model = None
        self.pred_df = None
        self.metrics_dict = None
        self.target = None
        self.target_class_map = {}
        self.columns_high_corr = None
        self.important_columns =None
        self.used_columns = None
        self.validation_percentage = validation_percentage
        self.cross_validation_k_folds = cross_validation_k_folds
        self.cross_validation_score = None 
        assert (self.validation_percentage<=0.9), "Validation % must be <=0.9"
        self.validation_df = None
        self.average_type = average_type
        self.problem_type = 'Classification'
        self.used_metric = 'f1'
        if feature_selection in ['correlation', 'importance']:
            self.feature_selection = feature_selection
        else:
            self.feature_selection = None

    def encode_target_column(self, train , target):
        for i, label in enumerate(train[target].unique()):
            self.target_class_map[label] = i
        train[target].replace(to_replace = self.target_class_map, inplace = True)
        return train


    def accuracy_history(self):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model,
            self.train_df[self.used_columns], 
            self.target_col,
            cv=10, 
            scoring='accuracy', 
            n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 7)
            )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        title = str(self.model)[:-2] +  ' learning curves'

        data = {
            'x':train_sizes,
            'y1':train_scores_mean,
            'y2':test_scores_mean,
            'title':title,
        }
        return data

    def plot(self):
        data = self.accuracy_history()
        plt.figure(figsize=(20,10))
        plt.title(data['title'], fontsize = 22)
        plt.plot(data['x'], data['y1'],color = 'blue', label = 'train accuarcy')
        plt.plot(data['x'], data['y2'],color = 'orange', label = 'test accuarcy')
        plt.legend()
        plt.xlabel('Sample size', fontsize = 18)
        plt.ylabel('Accuracy', fontsize = 18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()

    def gen_pred_df(self, df):
        preds = self.model.predict(df[self.used_columns])
        df[self.target] = preds
        # assign to self.pred_df only if inputs the test df
        if df is self.test_df:
            # reverse class mapping
            rev_class_map = dict((v,k) for k,v in self.target_class_map.items())
            df[self.target].replace(to_replace = rev_class_map, inplace = True)
            self.pred_df = df 
        else: # if input is validation
            return df

    def gen_metrics_dict(self):
        predected = self.gen_pred_df(self.validation_df)
        y_true = self.true_values
        y_pred = predected[self.target]
        # Handling different classification types (binary, multiclass)
        number_of_classes = len(self.target_col.unique())
        assert (number_of_classes >= 2) , "Target column contains less than 2 classes."
        if number_of_classes == 2: # binary classification
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            h_loss = hamming_loss(y_true, y_pred)
        elif number_of_classes > 2: # multiclass classification
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average = self.average_type)
            pre = precision_score(y_true, y_pred, average = self.average_type)
            recall = recall_score(y_true, y_pred, average = self.average_type)
            h_loss = hamming_loss(y_true, y_pred)

        dict_metrics = {
            "accuracy": acc,
            "f1": f1,
            "precision": pre,
            "recall": recall,
            "hamming_loss": h_loss,
            "cross_validation_score":self.cross_validation_score,
        }
        self.metrics_dict = dict_metrics
    
    def run(self):
        train , test = self.drop_null_data()
        target , dt , target_list = self.detect_target_data( train , test)
        df = self.concat_dataframes(target, train , test, target_list)
        df = self.drop_null_columns(df)
        cat_colmns , num_colmns = self.classify_columns(df, target)
        columns_has_2 , columns_has_3to7 , columns_to_drop = self.classify_categorical_columns(df , cat_colmns)
        df = self.fill_null_values(df, cat_colmns, num_colmns)
        df = self.encode_categorical_columns(df, columns_has_2, columns_has_3to7 , columns_to_drop)
        train_n , test_n = self.split_dataframes(df, target, dt)
        train_n = self.encode_target_column(train_n,target)
        self.split_train_validation(train_n,test_n,target)
        self.train_the_model()
        self.gen_pred_df(self.test_df)
        self.gen_metrics_dict()