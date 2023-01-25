
# blitzml

Automate machine learning piplines rapidly


## How to install


```bash
pip install blitzml
```


## Quick Usage

```python
from blitzml.tabular import Classification
import pandas as pd

# prepare your dataframes
train_df = pd.read_csv("auxiliary/datasets/banknote/train.csv")
test_df = pd.read_csv("auxiliary/datasets/banknote/test.csv")
ground_truth_df = pd.read_csv("auxiliary/datasets/banknote/ground_truth.csv")

# create the pipeline with a certain classifier
auto = Classification(train_df, test_df, ground_truth_df, classifier = 'RF', n_estimators = 50)

# first perform data preprocessing
auto.preprocess()
# second train the model
auto.train_the_model()


# After training the model we can generate the following:
auto.gen_pred_df()
auto.gen_metrics_dict()

# Then you can get their values using:
pred_df = auto.pred_df
metrics_dict = auto.metrics_dict

print(pred_df.head())
print(metrics_dict)
```


## Available Classifiers

- Random Forest 'RF'
- LinearDiscriminantAnalysis 'LDA'
- Support Vector Classifier 'SVC'

`The possible arguments for each model can be found in the `[sklearn docs](https://scikit-learn.org/stable/user_guide.html)

## Working with a custom classifier

```python
# create the pipeline with a custom classifier
# instead of specifying a classifier name we pass "custom" to the classifier argument.
auto = Classification(
    train_df,
    test_df,
    ground_truth_df,
    classifier="custom", 
    class_name = "classifier",
    file_path = "auxiliary/scripts/dummy.py"
)
```
## Smart Feature Selection

```python

```

## Development

- Clone the repo
- run `pip install virtualenv`
- run `python -m virtualenv venv`
- run `. ./venv/bin/activate` on UNIX based systems or `. ./venv/Scripts/activate.ps1` if on windows
- run `pip install -r requirements.txt`
- run `pre-commit install`
