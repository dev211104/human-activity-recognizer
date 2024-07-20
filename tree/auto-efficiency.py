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

max_depths = [5,6,7,8,9,10,11,12]
maes_t = []
for depth in max_depths:
    decision_tree = DecisionTree('mse',depth,2)
    decision_tree.fit(training_data,training_labels)
    pred_t = decision_tree.predict(testing_data)
    pred_tr = decision_tree.predict(training_data)
    maes_t.append(mae(pred_t,testing_labels))
print()
print(f'Mean Absolute Error using our decision tree:-\n')
print(maes_t)
print()
print("*"*100)
print()

from sklearn.tree import DecisionTreeRegressor


maes_s = []
for depths in max_depths:
    # Create a decision tree model
    decision_tree_model = DecisionTreeRegressor(random_state=42,max_depth=depths)

    # Fit the model
    decision_tree_model.fit(training_data, training_labels)

    # Make predictions on the testing data
    predictions = decision_tree_model.predict(testing_data)
    maes_s.append(mae(pd.Series(predictions),testing_labels))
print(f'Mean Absolute Error using our decision tree:-\n')
print(maes_s)
print()


'''
The key observation from the provided data is that the Mean Absolute Error (MAE) of our decision tree, 
is consistently higher across different maximum depths compared to the decision tree implemented using 
the scikit-learn decision tree. The MAE values decrease as the maximum depth increases for both 
implementations, indicating that a deeper tree tends to result in better predictions. However, even at
the same maximum depth, the scikit-learn decision tree consistently outperforms the manually implemented one, 
because of the way the have implemented it.
'''
