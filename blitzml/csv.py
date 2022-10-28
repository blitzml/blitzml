from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn import preprocessing
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix

# classifier can be ['RF', 'LDA', 'SVC']

class Pipeline:
	def __init__(self, dataset_path, ground_truth_path, output_folder_path, classifier='RF', n_estimators=100):
		self.dataset_path = dataset_path  # contains train.csv and test.csv
		self.ground_truth_path = ground_truth_path
		self.output_folder_path = output_folder_path
		self.classifier = classifier
		self.n_estimators = n_estimators
		self.train_df = None
		self.test_df = None
		self.target_col = None
		self.model = None
		self.metrics_dict = None

	def preprocess(self):
		train = pd.read_csv(self.dataset_path + 'train.csv') 
		test = pd.read_csv(self.dataset_path + 'test.csv')
		# drop duplicates
		train.drop_duplicates(inplace=True)
		# drop raws if contains null data in column have greater than 95% valid data (from train only)
		null_df = train.isnull().mean().to_frame()
		null_df['column'] = null_df.index
		null_df.index = np.arange(null_df.shape[0])
		null_cols = list(null_df[null_df[0].between(0.001,0.05)]['column'])
		for colmn in null_cols:
			null_index = list(train[train[colmn].isnull()].index)
			train.drop(index=null_index,axis=0,inplace=True)
		# get target data (column name,dtype,save values in list) 
		for col in train.columns:
			if col not in test.columns:
				target = col
		# get dtype
		dtype = train.dtypes.to_frame()
		dtype['column'] = dtype.index
		dtype.index = np.arange(dtype.shape[0])
		dt = str(dtype[dtype['column'] == target].iloc[0,0])  # target dtype
		if 'int' in dt:
			dt = int
		elif 'float' in dt:
			dt = float
		elif 'object' in dt:
			dt = str
		else:
			dt = "unknown"
		# save target list
		target_list = train[target]
		# concatinate datasets, first columns must be identical
		train.drop(columns=[target],inplace=True)
		train[target] = target_list
		train[target]=train[target].astype(str)
		test[target] = np.repeat('test_dataset',test.shape[0])
		df = pd.concat([train, test]) # concatinate datasets
		# drop columns na >= 25%
		null_df = df.isnull().mean().to_frame()
		null_df['column'] = null_df.index
		null_df.index = np.arange(null_df.shape[0])
		null_cols = list(null_df[null_df[0] >= .25]['column'])
		df.drop(columns=null_cols,inplace = True)
		# now we should know what is numerical columns and categorical columns
		dtype = df.dtypes.to_frame()
		dtype['column'] = dtype.index
		dtype.index = np.arange(dtype.shape[0])
		cat_colmns =[]
		num_colmns =[]
		columns_genres = [cat_colmns, num_colmns]
		num_values = ['float64','int64','uint8','int32','int8', 'int16', 'uint16', 'uint32', 'uint64','float_', 'float16', 'float32','int_','int','float']
		for i in range(len(dtype.column)):
			if 'object' in str(dtype.iloc[i,0]):
				cat_colmns.append(dtype.column[i])
			elif str(dtype.iloc[i,0]) in num_values:
				num_colmns.append(dtype.column[i])
		# remove target column from lists (not from dataframe)
		columns_genres = [cat_colmns,num_colmns]
		for genre in columns_genres:
			if target in genre:
				genre.remove(target)
		# drop columns has more than 7 unique values (from categorical columns)
		columns_has_2 = []
		columns_has_3to7 = []
		columns_to_drop = []
		for c_col in cat_colmns:
			if df[c_col].nunique() > 7:
				columns_to_drop.append(c_col)
			elif 3 <= df[c_col].nunique() <=7:
				columns_has_3to7.append(c_col)
			else:
				columns_has_2.append(c_col)
		# fillna in categorical columns
		df[cat_colmns] = df[cat_colmns].fillna('unknown')
		# fillna in numerical columns
		for column in num_colmns:
			df[column].fillna(value=df[column].mean(), inplace=True)
		# now we can drop without raising error
		df.drop(columns=columns_to_drop,inplace = True)
		# encode the categorical
		encoder = preprocessing.LabelEncoder()
		for col in columns_has_2:
			df[col] = encoder.fit_transform(df[col]) # encode columns has 2 unique values in the same column 0, 1
		# encode columns has 3-7 unique values
		for cat in columns_has_3to7:
			df = pd.concat([df, pd.get_dummies(df[cat], prefix=cat)], axis=1)
			df = df.drop([cat], axis=1)
		# split train and test 
		train_n = df[df[target] != "test_dataset"]
		test_n = df[df[target] == "test_dataset"].drop(target,axis = 1)
		try:
			train_n[target] = train_n[target].astype(dt)
		except:
			pass
		# assign processed dataframes
		self.train_df = train_n.drop(target, axis = 1)
		self.test_df  = test_n
		self.target_col = train_n[target]

	def train_the_model(self):
		X = self.train_df
		y = self.target_col

		if self.classifier == 'RF' :
			self.model = RandomForestClassifier(n_estimators = self.n_estimators)
			self.model.fit(X, y)

		elif self.classifier == 'LDA':
			self.model = LinearDiscriminantAnalysis()
			self.model.fit(X, y)

		elif self.classifier == 'SVC':
			self.model = SVC()
			self.model.fit(X, y)

	def output_model_file(self):
		if self.model == None:
			self.train_the_model()
		with open( self.output_folder_path + '/' + self.classifier + '.pkl', 'wb') as f:
			joblib.dump(self.model, f)

			
	def gen_metrics_dict(self):
		# use self.model with the test_df and self.ground_truth_path 
	    preds = self.model.predict(self.test_df)
	    # save the output file submission.csv to the output folder
	    pred_df = pd.DataFrame({"PassengerId": self.test_df["PassengerId"].values, "Survived": preds })
	    pred_df.to_csv(self.output_folder_path+'/'+'submission.csv')
		# get metrics dict
	    submission = pd.read_csv(self.output_folder_path+'/'+'submission.csv')
	    ground_truth_df = pd.read_csv(self.ground_truth_path)
	    x = ground_truth_df['Survived']
	    y = submission['Survived']
	    acc = accuracy_score(x, y)
	    f1 = f1_score(x, y)
	    pre = precision_score(x, y)
	    recall = recall_score(x, y)
	    tn, fp, fn, tp = confusion_matrix(x, y).ravel()
	    specificity = tn / (tn+fp)

	    dict_metrics = {'Accuracy': acc, 'f1': f1, 'Precision': pre, 'Recall': recall, 'Specificity': specificity}

	    # assign the resulting dictionary to self.metrics_dict
	    self.metrics_dict = dict_metrics