import sys
import importlib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_validate , cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression

class supervised():
    
    def get_custom_algorithm(self):
        assert(
                self.class_name != "None" and self.file_path != "None"
            ), "Didn't provide the custom algorithm arguments!"

        # load module using a class_name and a file_path
        spec = importlib.util.spec_from_file_location(self.class_name, self.file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.class_name] = module
        spec.loader.exec_module(module)

        # returns the class from the loaded module
        return module.__dict__[self.class_name] 
    
    def drop_null_data(self):
        train = self.train_df.copy()
        test = self.test_df.copy()
        # drop duplicates
        train.drop_duplicates(inplace=True)
        # drop raws if contains null data in column have greater than 95% valid data (from train only)
        null_df = train.isnull().mean().to_frame()
        null_df["column"] = null_df.index
        null_df.index = np.arange(null_df.shape[0])
        null_cols = list(null_df[(null_df[0] > 0) & (null_df[0] < 0.05) ]["column"])
        for colmn in null_cols:
            null_index = list(train[train[colmn].isnull()].index)
            train.drop(index=null_index, axis=0, inplace=True)
        return train , test
    
    def detect_target_data(self , train , test):
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
        return target , dt , target_list
    
    def concat_dataframes(self, target, train , test, target_list):
        # concatinate datasets, first columns must be identical
        train.drop(columns=[target], inplace=True)
        train[target] = target_list
        train[target] = train[target].astype(str)
        test[target] = np.repeat("test_dataset", test.shape[0])
        df = pd.concat([train, test])  # concatinate datasets
        return df
    
    def drop_null_columns(self, df ):
        # drop columns na >= 25%
        null_df = df.isnull().mean().to_frame()
        null_df["column"] = null_df.index
        null_df.index = np.arange(null_df.shape[0])
        null_cols = list(null_df[null_df[0] >= 0.25]["column"])
        df.drop(columns=null_cols, inplace=True)
        return df
    
    def classify_columns(self , df, target):
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
        return cat_colmns , num_colmns

    def classify_categorical_columns(self, df , cat_colmns):
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
        return columns_has_2 , columns_has_3to7 , columns_to_drop
    
    def fill_null_values(self, df , cat_colmns ,num_colmns):
        # fillna in categorical columns
        df[cat_colmns] = df[cat_colmns].fillna("unknown")
        # fillna in numerical columns
        for column in num_colmns:
            df[column].fillna(value=df[column].mean(), inplace=True)
        return df
    
    def encode_categorical_columns(self, df , columns_has_2 , columns_has_3to7,columns_to_drop):
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
        return df
    
    def split_dataframes(self , df , target , dt):
        # split train and test
        train_n = df[df[target] != "test_dataset"]
        test_n = df[df[target] == "test_dataset"].drop(target, axis=1)
        try:
            train_n[target] = train_n[target].astype(dt)
        except:
            pass
        return train_n , test_n
    
    def split_train_validation(self,train_n,test_n, target):
        validation_percentage = self.validation_percentage
        validation_index = int(len(self.train_df) * (1 - validation_percentage))
        # assign processed dataframes
        self.target_col = train_n.iloc[:validation_index,:][target]
        self.true_values = train_n.iloc[validation_index:,:][target]
        self.validation_df = train_n.iloc[validation_index:,:].drop(target, axis=1)
        self.train_df = train_n.iloc[:validation_index,:].drop(target, axis=1)
        self.test_df = test_n
        self.target = target
    
    def select_high_correlation(self):
        train_n = self.train_df
        target = self.target
        train_n[target] = self.target_col
        # classify columns by correlation
        corr_df = train_n.corr()
        # drop target raw 
        corr_df.drop(index=target,axis=0, inplace=True)
        # calculate corelation ref
        corr_ref = round(np.percentile(abs(corr_df[target].fillna(0)),33),4)
        columns_high_corr = list(corr_df[(corr_df[target] >= corr_ref)].index) + list(
            corr_df[(corr_df[target] <= -corr_ref)].index
        )
        self.columns_high_corr = columns_high_corr
    
    def select_important_features(self, train, target_column):
        fs = SelectKBest(score_func=mutual_info_regression, k='all')
        fs.fit(train,target_column)
        fs_df = pd.DataFrame({
        "feature":list(fs.feature_names_in_),
        "importance":list(fs.scores_)
        })
        important_columns= list(fs_df[fs_df['importance'] >= 0.1]['feature'])
        self.important_columns = important_columns

    def used_cols(self, train):
        if self.feature_selection == 'correlation':
            self.select_high_correlation()
            if self.columns_high_corr:
                self.used_columns = self.columns_high_corr
            else:
                print('there are no high correlation columns, the model used all features')
                self.used_columns = list(train.columns)
        elif self.feature_selection == 'importance':
            self.select_important_features(self.train_df ,self.target_col )
            if self.important_columns:
                self.used_columns = self.important_columns
            else:
                print('there are no important columns, the model used all features')
                self.used_columns = list(train.columns)
        elif self.feature_selection == None:
            self.used_columns = list(train.columns)
    
    def favorable_algorithm(self):
        x_train = self.train_df[self.used_columns]
        y_train  = self.target_col
        results = pd.DataFrame(columns=["algorithm", self.used_metric])
        for name, clf in self.algorithms_map.items():
            model = clf()
            cv_results = cross_validate(
                model, x_train , y_train , cv=10,
                scoring=[self.used_metric]
            )
            results = results.append({
                "algorithm": name,
                f"Avg_{self.used_metric}": cv_results[f'test_{self.used_metric}'].mean(),
            }, ignore_index=True)
            
        results = results.sort_values(f"Avg_{self.used_metric}", ascending=False)
        return self.algorithms_map[results.iloc[0,:]['algorithm']]
    
    def train_the_model(self):
        self.used_cols(self.train_df)
        # use high correlation columns only in training
        X = self.train_df[self.used_columns]
        y = self.target_col
        if self.algorithm == "custom":
            algorithm = self.get_custom_algorithm()
        elif self.algorithm == 'auto':
            algorithm = self.favorable_algorithm()
        else:
            algorithm = self.algorithms_map[self.algorithm]
        self.model = algorithm(**self.kwargs)
        # performing cross validation on the selected model and features
        if self.cross_validation_k_folds > 1:
            temp_model = self.model
            self.cross_validation_score = cross_val_score(self.model, X, y, cv=self.cross_validation_k_folds, scoring='neg_root_mean_squared_error')
            # reverse scaling
            if self.problem_type == 'Regression':
                self.cross_validation_score = self.scaler.inverse_transform(np.asarray(self.cross_validation_score).reshape(-1, 1)).reshape(1, -1)[0]
            self.model = temp_model

        self.model.fit(X, y)