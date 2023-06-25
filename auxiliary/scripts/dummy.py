import random

class classifier:
	def __init__ (self):
		pass
	def fit (self, X, y):
		pass
	def predict(self, X):
		target_values = [0, 1]
		return [random.choice(target_values) for x in range(len(X))]

class regressor:
	def __init__ (self):
		pass
	def fit (self, X, y):
		pass
	def predict(self, X):
		target_values = [0, 1]
		return [random.choice(target_values) for x in range(len(X))]

class clustering:
	def __init__ (self, n_clusters = 8):
		self.n_clusters = n_clusters
	def fit (self, X):
		pass
	def predict(self, X):
		target_values = [i for i in range(self.n_clusters)]
		return [random.choice(target_values) for x in range(len(X))]