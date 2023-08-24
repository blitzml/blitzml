import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
# ignore pandas warnings 
import warnings
warnings.filterwarnings('ignore')
# metrics imports 
from math import sqrt
from sklearn.metrics import (
    mean_squared_error, 
    r2_score ,
    mean_absolute_error
)
from ._supervised import Supervised_ML


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor

# preprocessing and cross validation
from sklearn.model_selection import learning_curve

class Regression(Supervised_ML):
    """
    Parameters:
        :param kwargs: is the regressor arguments
    """

    algorithms_map = {
        "LR": LinearRegression,
        "RDG": Ridge,
        "LSS": Lasso,
        "MLP": MLPRegressor,
        "GB": GradientBoostingRegressor,
        "DT": DecisionTreeRegressor,
        "RF": RandomForestRegressor,
        "KNN": KNeighborsRegressor,
        "SVR": SVR,
        "GPR": GaussianProcessRegressor,
    }

    def __init__(self,
                 train_df,
                 test_df,
                 algorithm="RF",
                 class_name = "None",
                 file_path = "None",
                 feature_selection = "none",
                 validation_percentage = 0.1,
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
            ), "Unsupported regressor provided"
            self.algorithm = algorithm

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
        self.true_values = None
        self.validation_percentage = validation_percentage
        self.cross_validation_k_folds = cross_validation_k_folds
        self.cross_validation_score = None 
        assert (self.validation_percentage<=0.9), "Validation % must be <=0.9"
        self.validation_df = None
        self.scaler = None
        self.problem_type = 'Regression'
        self.used_metric = 'r2'
        if feature_selection in ['correlation', 'importance']:
            self.feature_selection = feature_selection
        else:
            self.feature_selection = None

    def scale_target_column(self, train ,target):
        self.scaler = preprocessing.MinMaxScaler()
        train[target] = self.scaler.fit_transform(np.asarray(train[target]).reshape(-1, 1)).reshape(1, -1)[0]
        self.train_df = train

    def rmse_history(self):
        train_sizes, train_scores, test_scores = learning_curve(
            self.model,
            self.train_df[self.used_columns], 
            self.target_col,
            cv=5, 
            scoring='neg_root_mean_squared_error', 
            n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 7)
            )
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        title = str(self.model)[:-2] +  ' learning curves'

        data = {
            'x':train_sizes,
            'y1':train_scores_mean,
            'y2':test_scores_mean,
            'title':title,
        }
        return data
    
    def plot(self):
        data = self.rmse_history()
        plt.figure(figsize=(20,10))
        plt.title(data['title'], fontsize = 22)
        plt.plot(data['x'], data['y1'],color = 'blue', label = 'train RMSE')
        plt.plot(data['x'], data['y2'],color = 'orange', label = 'test RMSE')
        plt.legend()
        plt.xlabel('Sample size', fontsize = 18)
        plt.ylabel('Root mean squared error', fontsize = 18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()

    def gen_pred_df(self, df):
        preds = self.model.predict(df[self.used_columns])
        df[self.target] = preds
        # assign to self.pred_df only if inputs the test df
        if df is self.test_df:
            # reverse scaling
            df[self.target] = self.scaler.inverse_transform(np.asarray(df[self.target]).reshape(-1, 1)).reshape(1, -1)[0]
            self.pred_df = df 
        else: # if input is validation
            return df

    def gen_metrics_dict(self):
        predicted = self.gen_pred_df(self.validation_df)
        y_pred = predicted[self.target]
        y_true = self.true_values

        # inverse scaling
        y_pred = self.scaler.inverse_transform(np.asarray(y_pred).reshape(-1, 1)).reshape(1, -1)[0]
        y_true = self.scaler.inverse_transform(np.asarray(y_true).reshape(-1, 1)).reshape(1, -1)[0]

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        dict_metrics = {
            "r2_score": r2,
            "mean_squared_error": mse,
            "root_mean_squared_error": rmse,
            "mean_absolute_error" : mae,
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
        self.scale_target_column(train_n,target)
        self.split_train_validation(train_n,test_n,target)
        self.train_the_model()
        self.gen_pred_df(self.test_df)
        self.gen_metrics_dict()