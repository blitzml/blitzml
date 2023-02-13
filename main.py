import pandas as pd
from blitzml.tabular import Classification

# prepare your dataframes
dataset = 'titanic'
train_df = pd.read_csv(f"auxiliary/datasets/{dataset}/train.csv")
test_df = pd.read_csv(f"auxiliary/datasets/{dataset}/test.csv")

# create the pipeline
auto = Classification(
    train_df,
    test_df,
    classifier="MLP",
    feature_selection = "correlation"
)

# first perform data preprocessing
auto.preprocess()
# second train the model
auto.train_the_model()

# After training the model we can generate the following:
auto.gen_pred_df(auto.test_df)
auto.gen_metrics_dict()

# Then you can get their values using:
pred_df = auto.pred_df
metrics_dict = auto.metrics_dict

print(pred_df.head())
print(metrics_dict)
