import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

data = data.drop("car name", axis=1)
data = data.replace("?", np.nan)
data = data.dropna()
data = data.drop_duplicates()

# Now the data should be clean
data1 = data.reset_index(drop=True)
data1['horsepower'] = pd.to_numeric(data1['horsepower'], errors='coerce')

data1 = data1.drop("weight",axis=1)
data1 = data1.drop("origin",axis=1)
data1 = data1.drop("displacement",axis=1)
data1 = data1.drop("model year",axis=1)

features = list(data1.columns)
features.remove('mpg')


train_data = data1[:274]
test_data = data1[274:].reset_index(drop=True)

training_data = train_data[features]
training_labels = train_data['mpg'].reset_index(drop=True)

testing_data = test_data[features]
testing_labels = test_data['mpg'].reset_index(drop=True)
# training_labels = np.floor(training_labels)
# testing_labels = np.floor(testing_labels)

decision_tree = DecisionTree('mse',20)

decision_tree.fit(training_data,training_labels)
decision_tree.plot()
# print(decision_tree.dictionary)

test_labels = decision_tree.predict(testing_data)
