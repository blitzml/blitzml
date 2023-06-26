
<div align="center">
<img src="auxiliary/docs/logo.png" alt="BlitzML" width="400"/>

### **Automate machine learning pipelines rapidly**


<div align="left">

- [Install BlitzML](#install-blitzml)
- [Classification](#classification)
- [Regression](#regression)
- [Time-Series](#time-series)
- [Clustering](#clustering)



# Install BlitzML  


```bash
pip install blitzml
```


# Classification

```python
from blitzml.tabular import Classification
import pandas as pd

# prepare your dataframes
train_df = pd.read_csv("auxiliary/datasets/banknote/train.csv")
test_df = pd.read_csv("auxiliary/datasets/banknote/test.csv")

# create the pipeline
auto = Classification(train_df, test_df, classifier = 'RF', n_estimators = 50)

# first perform data preprocessing
auto.preprocess()
# second train the model
auto.train_the_model()

# After training the model we can generate:
auto.gen_pred_df(auto.test_df)
auto.gen_metrics_dict()

# We can get their values using:
pred_df = auto.pred_df
metrics_dict = auto.metrics_dict

print(pred_df.head())
print(metrics_dict)
```


## Available Classifiers

- Random Forest 'RF' 
- LinearDiscriminantAnalysis 'LDA' 
- Support Vector Classifier 'SVC' 
- KNeighborsClassifier 'KNN' 
- GaussianNB 'GNB' 
- LogisticRegression 'LR'
- AdaBoostClassifier 'AB'
- GradientBoostingClassifier 'GB'
- DecisionTreeClassifier 'DT'
- MLPClassifier 'MLP'


## **Parameters**
**classifier**  
options: {'RF','LDA','SVC','KNN','GNB','LR','AB','GB','DT','MLP', 'auto', 'custom'}, default = 'RF'  
`auto: selects the best scoring classifier based on f1-score`  
`custom: enables providing a custom classifier through *file_path* and *class_name* parameters`  
**file_path**  
when using 'custom' classifier, pass the path of the file containing the custom class, default = 'none'  
**class_name**  
when using 'custom' classifier, pass the class name through this parameter, default = 'none'  
**feature_selection**  
options: {'correlation', 'importance', 'none'}, default = 'none'  
`correlation: use feature columns with the highest correlation with the target`  
`importance: use feature columns that are important for the model to predict the target`  
`none: use all feature columns`  
**validation_percentage**  
value determining the validation split percentage (value from 0 to 1), default = 0.1  
**average_type**  
when performing multiclass classification, provide the average type for the resulting metrics, default = 'macro'  
**cross_validation_k_folds**  
number of k-folds for cross validation, if 1 then no cv will be performed, default = 1  
****kwargs**  
optional parameters for the chosen classifier. you can find available parameters in the [sklearn docs](https://scikit-learn.org/stable/user_guide.html)  
## **Attributes**  
**train_df**  
the preprocessed train dataset (after running `Classification.preprocess()`)  
**test_df**  
the preprocessed test dataset (after running `Classification.preprocess()`)  
**model**  
the trained model (after running `Classification.train_the_model()`)  
**pred_df**  
the prediction dataframe (test_df + predicted target) (after running `Classification.gen_pred_df(Classification.test_df)`)  
**metrics_dict**  
the validation metrics (after running `Classification.gen_metrics_dict()`)  
{  
    "accuracy": acc,  
    "f1": f1,  
    "precision": pre,  
    "recall": recall,  
    "hamming_loss": h_loss,  
    "cross_validation_score":cv_score, `returns None if cross_validation_k_folds==1`  
}   
## **Methods**  
**preprocess()**  
perform preprocessing on train_df and test_df  
**train_the_model()**  
train the chosen classifier on the train_df  
**accuracy_history()**  
accuracy scores when varying the sampling size of the train_df (after running `Classification.train_the_model()`).  
*returns:*  
{  
    'x':train_df_sample_sizes,  
    'y1':train_scores_mean,  
    'y2':test_scores_mean,  
    'title':title  
}  
**gen_pred_df(test_df)**  
generates the prediction dataframe and assigns it to the `pred_df` attribute  
**gen_metrics_dict()**  
generates the validation metrics and assigns it to the `metrics_dict`  
**run()**  
a shortcut that runs the following methods:  
preprocess()  
train_the_model()  
gen_pred_df(Classification.test_df)  
gen_metrics_dict()  
# Regression  

```python
from blitzml.tabular import Regression
import pandas as pd

# prepare your dataframes
train_df = pd.read_csv("auxiliary/datasets/house prices/train.csv")
test_df = pd.read_csv("auxiliary/datasets/house prices/test.csv")

# create the pipeline
auto = Regression(train_df, test_df, regressor = 'RF')

# first perform data preprocessing
auto.preprocess()
# second train the model
auto.train_the_model()

# After training the model we can generate:
auto.gen_pred_df(auto.test_df)
auto.gen_metrics_dict()

# We can get their values using:
pred_df = auto.pred_df
metrics_dict = auto.metrics_dict

print(pred_df.head())
print(metrics_dict)
```


## Available Regressors

- Random Forest 'RF'
- Support Vector Regressor 'SVR'
- KNeighborsRegressor 'KNN'
- Lasso Regressor 'LSS'
- LinearRegression 'LR'
- Ridge Regressor 'RDG'
- GaussianProcessRegressor 'GPR'
- GradientBoostingRegressor 'GB'
- DecisionTreeRegressor 'DT'
- MLPRegressor 'MLP'

## **Parameters**
**regressor**  
options: {'RF','SVR','KNN','LSS','LR','RDG','GPR','GB','DT','MLP', 'auto', 'custom'}, default = 'RF'  
`auto: selects the best scoring regressor based on r2 score`  
`custom: enables providing a custom regressor through *file_path* and *class_name* parameters`  
**file_path**  
when using 'custom' regressor, pass the path of the file containing the custom class, default = 'none'  
**class_name**  
when using 'custom' regressor, pass the class name through this parameter, default = 'none'  
**feature_selection**  
options: {'correlation', 'importance', 'none'}, default = 'none'  
`correlation: use feature columns with the highest correlation with the target`  
`importance: use feature columns that are important for the model to predict the target`  
`none: use all feature columns`  
**validation_percentage**  
value determining the validation split percentage (value from 0 to 1), default = 0.1  
**cross_validation_k_folds**  
number of k-folds for cross validation, if 1 then no cv will be performed, default = 1  
****kwargs**  
optional parameters for the chosen regressor. you can find available parameters in the [sklearn docs](https://scikit-learn.org/stable/user_guide.html)  
## **Attributes**  
**train_df**  
the preprocessed train dataset (after running `Regression.preprocess()`)  
**test_df**  
the preprocessed test dataset (after running `Regression.preprocess()`)   
**model**  
the trained model (after running `Regression.train_the_model()`)  
**pred_df**  
the prediction dataframe (test_df + predicted target) (after running `Regression.gen_pred_df(Regression.test_df)`)  
**metrics_dict**  
the validation metrics (after running `Regression.gen_metrics_dict()`)  
{  
    "r2_score": r2,  
    "mean_squared_error": mse,  
    "root_mean_squared_error": rmse,  
    "mean_absolute_error" : mae,  
    "cross_validation_score":cv_score, `returns None if cross_validation_k_folds==1`  
}  
## **Methods**  
**preprocess()**  
perform preprocessing on train_df and test_df  
**train_the_model()**  
train the chosen regressor on the train_df  
**RMSE_history()**  
RMSE scores when varying the sampling size of the train_df (after running `Regression.train_the_model()`).  
*returns:*  
{  
    'x':train_df_sample_sizes,  
    'y1':train_scores_mean,  
    'y2':test_scores_mean,  
    'title':title  
}  
**gen_pred_df(test_df)**  
generates the prediction dataframe and assigns it to the `pred_df` attribute  
**gen_metrics_dict()**  
generates the validation metrics and assigns it to the `metrics_dict`  
**run()**  
a shortcut that runs the following methods:  
preprocess()  
train_the_model()  
gen_pred_df(Regression.test_df)  
gen_metrics_dict()  
# Time-series
time series is a particular problem of Regression, but time series have some additional functions:
- stationary test (IsStationary()). 
- convert to stationary.
- reverse predicted.

and the dataset must have a DateTime column, even if the DataType of this column is Object.
```python
from blitzml.tabular import TimeSeries 
import pandas as pd

# prepare your dataframes
train_df = pd.read_csv("train_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

# create the pipeline
auto = TimeSeries(train_df, test_df, regressor = 'RF')

# Perform the entire process:
auto.run()

# We can get their values using:
pred_df = auto.pred_df
metrics_dict = auto.metrics_dict

print(pred_df.head())
print(metrics_dict)
```
# Clustering 

```python
from blitzml.unsupervised import Clustering
import pandas as pd

# prepare your dataframe
train_df = pd.read_csv("auxiliary/datasets/customer personality/train.csv")

# create the pipeline
auto = Clustering(train_df, clustering_algorithm = 'KM')

# first perform data preprocessing
auto.preprocess()
# second train the model
auto.train_the_model()

# After training the model we can generate:
auto.gen_pred_df()
auto.gen_metrics_dict()

# We can get their values using:
print(auto.pred_df.head())
print(auto.metrics_dict)
```


## Available Clustering Algorithms 

- K-Means 'KM' 
- Affinity Propagation 'AP' 
- Agglomerative Clustering 'AC' 
- Mean Shift 'MS' 
- Spectral Clustering 'SC' 
- Birch 'Birch' 
- Bisecting K-Means 'BKM' 
- OPTICS 'OPTICS' 
- DBSCAN 'DBSCAN' 

## **Parameters** 
**clustering_algorithm**  
options: {"KM", "AP", "AC", "MS", "SC", "Birch", "BKM", "OPTICS", "DBSCAN", 'auto', 'custom'}, default = 'KM' 
`auto: selects the best scoring clustering algorithm based on silhouette score` 
`custom: enables providing a custom clustering algorithm through *file_path* and *class_name* parameters` 
**file_path** 
when using 'custom' clustering_algorithm, pass the path of the file containing the custom class, default = 'none'   
**class_name**
when using 'custom' clustering_algorithm, pass the class name through this parameter, default = 'none' 
**feature_selection** 
options: {'importance', 'none'}, default = 'none' 
`importance: use feature columns that are important for the model to predict the target` 
`none: use all feature columns` 
****kwargs** 
optional parameters for the chosen clustering_algorithm. you can find available parameters in the [sklearn docs](https://scikit-learn.org/stable/user_guide.html) 
## **Attributes** 
**train_df** 
the preprocessed train dataset (after running `Clustering.preprocess()`)  
**model** 
the trained model (after running `Clustering.train_the_model()`) 
**pred_df** 
the prediction dataframe (test_df + predicted target) (after running `Clustering.gen_pred_df()`) 
**metrics_dict** 
the validation metrics (after running `Clustering.gen_metrics_dict()`) 
{ 
    "silhouette_score": sil_score, 
    "calinski_harabasz_score": cal_har_score, 
    "davies_bouldin_score": dav_boul_score, 
    "n_clusters" : n 
} 
## **Methods** 
**preprocess()** 
perform preprocessing on train_df  
**train_the_model()** 
train the chosen clustering algorithm on the train_df 
**clustering_visualization()** 
2-d visualization of the data points with its corresponding labels  (after doing dimensionality reduction using Principal Componenet Analysis). 
*returns:* 
{ 
    'principal_component_1':pc1, 
    'principal_component_2':pc2, 
    'cluster_labels':labels, 
    'title':title 
} 
**gen_pred_df()** 
generates the prediction dataframe and assigns it to the `pred_df` attribute 
**gen_metrics_dict()** 
generates the clustering metrics and assigns it to the `metrics_dict`  
**run()** 
a shortcut that runs the following methods: 
preprocess() 
train_the_model() 
gen_pred_df() 
gen_metrics_dict() 
## Development  

- Clone the repo  
- run `pip install virtualenv`
- run `python -m virtualenv venv`
- run `. ./venv/bin/activate` on UNIX based systems or `. ./venv/Scripts/activate.ps1` if on windows
- run `pip install -r requirements.txt`
- run `pre-commit install`
