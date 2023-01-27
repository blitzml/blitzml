
# blitzml

Automate machine learning pipelines rapidly


## How to install


```bash
pip install blitzml
```


## Usage

```python
from blitzml.tabular import Classification
import pandas as pd

# prepare your dataframes
train_df = pd.read_csv("auxiliary/datasets/banknote/train.csv")
test_df = pd.read_csv("auxiliary/datasets/banknote/test.csv")
ground_truth_df = pd.read_csv("auxiliary/datasets/banknote/ground_truth.csv")

# create the pipeline
auto = Classification(train_df, test_df, ground_truth_df, classifier = 'RF', n_estimators = 50)

# first perform data preprocessing
auto.preprocess()
# second train the model
auto.train_the_model()


# After training the model we can generate:
auto.gen_pred_df()
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

The possible arguments for each model can be found in the [sklearn docs](https://scikit-learn.org/stable/user_guide.html)

## Working with a custom classifier

```python
# instead of specifying a classifier name, we pass "custom" to the classifier argument.
auto = Classification(
    train_df,
    test_df,
    ground_truth_df,
    classifier = "custom", 
    class_name = "classifier",
    file_path = "auxiliary/scripts/dummy.py",
    feature_selection = "importance"
)
```
## Smart Feature Selection

```python
# to filter used columns by correlation with target column
auto = Classification(
    '''
    params
    '''
    feature_selection = "correlation"
)
# to filter used columns by feature importance (boruta)
auto = Classification(
    '''
    params
    '''
    feature_selection = "importance"
)
# to use all columns
auto = Classification(
    '''
    params
    '''
    feature_selection = "None"
)
```
## Additional features
### • Preprocessing a dataset
```python
# After executing
auto.preprocess()
# You can access the processed datasets via
processed_train_df = auto.train_df
processed_test_df = auto.test_df
```
### • Less coding
```python
# Instead of
auto.preprocess()
auto.train_the_model()
auto.gen_pred_df()
auto.gen_metrics_dict()
# You can simply use
auto.run()
```

## Development

- Clone the repo
- run `pip install virtualenv`
- run `python -m virtualenv venv`
- run `. ./venv/bin/activate` on UNIX based systems or `. ./venv/Scripts/activate.ps1` if on windows
- run `pip install -r requirements.txt`
- run `pre-commit install`
