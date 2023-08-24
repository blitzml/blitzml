import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.errors import ParserError 
from datetime import datetime
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
from statsmodels.tsa.stattools import adfuller



from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor


class TimeSeries(Supervised_ML):
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
        self.columns_high_corr = None
        self.important_columns =None
        self.used_columns = None
        self.true_values = None
        self.validation_percentage = validation_percentage
        self.cross_validation_k_folds = cross_validation_k_folds
        self.cross_validation_score = None 
        assert (self.validation_percentage<=0.9), "Validation % must be <=0.9"
        self.validation_df = None
        self.stationary = None
        self.problem_type = 'Time-Series'
        self.used_metric = 'r2'
        if feature_selection in ['correlation', 'importance']:
            self.feature_selection = feature_selection
        else:
            self.feature_selection = None

    def detect_date_column(self, df, cat_colmns):
        for c in df.columns[df.dtypes=='object']: 
            try:
                df[c]=pd.to_datetime(df[c])
            except (ParserError,ValueError): 
                pass
        date_col_name = df.columns[df.dtypes == 'datetime64[ns]'][0]
        try:
            cat_colmns.remove(date_col_name)
        except:
            pass
        df.index = df[date_col_name]
        return date_col_name , df
    
    def create_date_features(self, df, date_col_name):
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        return df.drop([date_col_name], axis = 1)


    def is_stationary(self,target_col):
        result = adfuller(target_col)
        p_value = result[1]
        if p_value > 0.05:
            self.stationary = False
            return False
        else:
            self.stationary = True
            return True

    def convert_to_stationary(self, target_col):
        target_head = target_col[0]
        options = {
            "diff": target_col.diff(),
            "log": np.log(target_col)
        }
        for option_name, transformed_col in options.items():
            if self.is_stationary(transformed_col.dropna()):
                print(f"{option_name} can convert data to stationary")
                self.stationary = True
                return transformed_col, option_name, target_head
        raise Exception(f"Sorry, the target column ({self.target}) isn't stationary and blitzML can't convert your data to stationary")


    def train_pred_visualization(self):

        x_train = self.train_target.index.astype(str)
        y_train = self.train_target
        x_pred = self.test_df.index.astype(str)
        y_pred= self.pred_df[self.target]     
        
        x_train = list(x_train)
        y_train = list(y_train)
        x_pred = list(x_pred)
        y_pred = list(y_pred)

        title = f'{self.target} overtime'

        data = {
        'x_train': x_train,
        'y_train': y_train,
        'x_pred':x_pred,
        'y_pred':y_pred,
        'title':title,
        }
        return data

    def convert_str_to_datetime(self, list_of_strings):
        list_of_dates =   [datetime.strptime(date_string, '%Y-%m-%d') for date_string in list_of_strings]
        return list_of_dates

    def plot(self):
        data = self.train_pred_visualization()
        plt.figure(figsize=(20,10))
        plt.title(data['title'], fontsize = 22)
        plt.plot(self.convert_str_to_datetime(data['x_train']), data['y_train'],color = 'blue', label = 'training data')
        plt.plot(self.convert_str_to_datetime(data['x_pred']), data['y_pred'],color = 'orange', label = 'predicted data')
        plt.legend()
        plt.xlabel('Date', fontsize = 18)
        plt.ylabel(self.target, fontsize = 18)
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.show()

    def gen_pred_df(self, df):
        preds = self.model.predict(df[self.used_columns])
        df[self.target] = preds
        # assign to self.pred_df only if inputs the test df
        if df is self.test_df:
            self.pred_df = df 
        else: # if input is validation
            return df

    def gen_metrics_dict(self):
        predicted = self.gen_pred_df(self.validation_df)
        y_pred = predicted[self.target]
        y_true = self.true_values 

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
    
    def reverse_predicted(self,target_col,option,target_head,predicted):
        if option == 'diff':
            tc = pd.concat([target_col,predicted])
            reversed_tc = pd.Series(target_head,index = tc.index)
            for i in range(1,len(tc)):
                reversed_tc[i] = reversed_tc[i-1] + tc[i]
            return reversed_tc[len(target_col):]
        
        elif option == 'log':
            return np.exp(predicted)
    
    def run(self):
        train , test = self.drop_null_data()
        target , dt , target_list = self.detect_target_data( train , test)
        df = self.concat_dataframes(target, train , test, target_list)
        df = self.drop_null_columns(df)
        cat_colmns , num_colmns = self.classify_columns(df, target)
        date_col_name , df = self.detect_date_column(df,cat_colmns)
        df = self.create_date_features(df, date_col_name)
        columns_has_2 , columns_has_3to7 , columns_to_drop = self.classify_categorical_columns(df , cat_colmns)
        df = self.fill_null_values(df, cat_colmns, num_colmns)
        df = self.encode_categorical_columns(df, columns_has_2, columns_has_3to7 , columns_to_drop)
        train_n , test_n = self.split_dataframes(df, target, dt)
        self.train_target = train_n[target]
        self.split_train_validation(train_n,test_n,target)
        if self.is_stationary(self.target_col):
            self.train_the_model()
            self.gen_pred_df(self.test_df)
            self.gen_metrics_dict()
        else:
            transformed_target,name,target_head = self.convert_to_stationary(self.train_target)
            transformed_target = transformed_target.fillna(0) # if diff function is used first raw will nan
            # transformed_val ==> transformed validation true values to compute metrics
            self.true_values = transformed_target[len(self.target_col):]
            # ransformed_target ==> stationary target column
            self.target_col = transformed_target[:len(self.target_col)] 

            self.train_the_model()
            self.gen_pred_df(self.test_df)
            self.gen_metrics_dict()
            predicted_reversed = self.reverse_predicted(transformed_target,name,target_head,self.pred_df[self.target])
            self.pred_df[self.target] = predicted_reversed