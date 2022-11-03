import random

class classifier:
	def __init__ (self):
		pass
	def fit (self, X, y):
		pass
	def predict(self, X):
		target_values = [0, 1]
		return [random.choice(target_values) for x in range(len(X))]