
# blitzml

Automate machine learning piplines rapidly


## How to install


```bash
pip install blitzml
```


## Usage

```python
from blitzml.tabular import Classification
import pandas as pd

# prepare your dataframes
train_df = pd.read_csv("auxiliary/data/train.csv")
test_df = pd.read_csv("auxiliary/data/test.csv")
ground_truth_df = pd.read_csv("auxiliary/ground_truth.csv")

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

`When using RF you can also provide the number of estimators`

`via n_estimators = 100 (default)`



## Development

- Clone the repo
- run `pip install virtualenv`
- run `python -m virtualenv venv`
- run `. ./venv/bin/activate` on UNIX based systems or `. ./venv/Scripts/activate.ps1` if on windows
- run `pip install -r requirements.txt`
- run `pre-commit install`
