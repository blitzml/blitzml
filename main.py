import pandas as pd

from blitzml.tabular import Classification

# prepare your dataframes
train_df = pd.read_csv("auxiliary/data/train.csv")
test_df = pd.read_csv("auxiliary/data/test.csv")
ground_truth_df = pd.read_csv("auxiliary/ground_truth.csv")

# create the pipeline with a certain classifier
auto = Classification(
    train_df, test_df, ground_truth_df, classifier="RF", n_estimators=50
)

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
