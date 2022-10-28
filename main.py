from blitzml.csv import Pipeline

dataset_folder = "auxiliary/data/"
ground_truth_path = "auxiliary/ground_truth.csv"
output_folder_path = "auxiliary/output/"

auto = Pipeline(dataset_folder, ground_truth_path, output_folder_path, classifier = 'RF', n_estimators = 50)

auto.preprocess()
auto.train_the_model()
auto.gen_metrics_dict()

metrics_dict = auto.metrics_dict
print(metrics_dict)