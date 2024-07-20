import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from itertools import product

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)

# Write the code for Q2 a) and b) below. Show your results.

# Q2 a)

X_train,X_test = X[:70],X[70:]
Y_train,Y_test = y[:70],y[70:]
x_train = pd.DataFrame(X_train,columns=['f1','f2'])
x_test = pd.DataFrame(X_test,columns=['f1','f2'])
y_train = pd.Series(Y_train,name='Y')
y_test = pd.Series(Y_test,name='Y')

max_depth = 10
depth_accuracys_test = []
for i in range(1,max_depth):
    Decision_tree = DecisionTree('mse',i)
    Decision_tree.fit(x_train,y_train)
    y_pred = pd.Series(Decision_tree.predict(x_test))
    depth_accuracys_test.append(accuracy(y_pred,y_test))
print(f'Accuracy at different depth:-\n')
print(depth_accuracys_test)
print("*"*100)
opt_depth = np.argmax(depth_accuracys_test)
print(f"Optimal depth of the tree is:-{opt_depth}")
Decision_tree = DecisionTree('mse',1)
Decision_tree.fit(x_train,y_train)
y_pred = Decision_tree.predict(x_test)
print(f"Plot of the decision tree at optimal depth:-")
Decision_tree.plot()

acc = accuracy(y_pred,y_test)
prec_0 = precision(y_pred,y_test,0)
prec_1 = precision(y_pred,y_test,1)
rec_0 = recall(y_pred,y_test,0)
rec_1 = recall(y_pred,y_test,1)

print(f"Accuracy at depth: {opt_depth} is:- {acc}")
print(f"Precision of class 0 at depth: {opt_depth} is:- {prec_0}")
print(f"Precision of class 1 at depth: {opt_depth} is:- {prec_1}")
print(f"Recall of class 0 at depth: {opt_depth} is:- {rec_1}")
print(f"Recall of class 1 at depth: {opt_depth} is:- {rec_1}")


max_depth = 10
depth_accuracys_train = []

for i in range(1,max_depth):
    Decision_tree = DecisionTree('mse',i)
    Decision_tree.fit(x_train,y_train)
    y_pred = pd.Series(Decision_tree.predict(x_train))
    depth_accuracys_train.append(accuracy(y_pred,y_train))
print(f"Accuracy of training set:-")
print(depth_accuracys_train)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_depth), depth_accuracys_train, marker='o', label='Training Accuracy')
plt.plot(range(1, max_depth), depth_accuracys_test, marker='o', label='Test Accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs Depth')
plt.legend()
plt.show()



# Q2 b)

# Define the number of folds (k)
k = 5

# Initialize lists to store predictions and accuracies
predictions = {}
accuracies = []

# Calculate the size of each fold
fold_size = len(X) // k

# Perform k-fold cross-validation
for i in range(k):
    # Split the data into training and test sets
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    test_set = X[test_start:test_end]
    test_labels = pd.Series(y[test_start:test_end],name='Y')
    
    training_set = np.concatenate((X[:test_start], X[test_end:]), axis=0)
    training_labels = np.concatenate((y[:test_start], y[test_end:]), axis=0)
    training_set = pd.DataFrame(training_set,columns=['f1','f2'])
    training_labels_1 = pd.Series(training_labels,name='Y')
    
    # Train the model
    dt_classifier = DecisionTree('mse',3)
    dt_classifier.fit(training_set, training_labels_1)
    
    # Make predictions on the validation set
    fold_predictions = dt_classifier.predict(pd.DataFrame(test_set,columns=['f1','f2']))
    
    # Calculate the accuracy of the fold
    fold_accuracy = np.mean(fold_predictions == test_labels)
    
    # Store the predictions and accuracy of the fold
    predictions[i] = fold_predictions
    accuracies.append(fold_accuracy)

# Print the predictions and accuracies of each fold
for i in range(k):
    print("Fold {}: Accuracy: {:.4f}".format(i+1, accuracies[i]))

# Define hyperparameters for decision tree
hyperparameters = {
    'max_depth': [1, 2, 3, 4, 5],
    'min_samples_split': [2, 3, 4, 5]
}

# Initialize lists to store results
outer_fold_accuracies = []
outer_fold_opt_depth = []
outer_fold_opt_minsamples = []
# Perform outer k-fold cross-validation
k = 5
outer_fold_size = len(X) // k

for outer_fold in range(k):
    # Split data into outer training and test sets
    test_start = outer_fold * outer_fold_size
    test_end = (outer_fold + 1) * outer_fold_size
    test_set = X[test_start:test_end]
    test_labels = y[test_start:test_end]

    train_set = np.concatenate((X[:test_start], X[test_end:]), axis=0)
    train_labels = np.concatenate((y[:test_start], y[test_end:]), axis=0)

    # Initialize variables for optimal hyperparameters and accuracy
    best_max_depth = None
    best_min_samples_split = None
    best_accuracy = 0.0

    # Perform inner k-fold cross-validation to find optimal hyperparameters
    inner_fold_size = len(train_set) // k

    for inner_fold in range(k):
        val_start = inner_fold * inner_fold_size
        val_end = (inner_fold + 1) * inner_fold_size
        val_set = train_set[val_start:val_end]
        val_labels = train_labels[val_start:val_end]
        training_set = np.concatenate((train_set[:val_start], train_set[val_end:]), axis=0)
        training_labels = np.concatenate((train_labels[:val_start], train_labels[val_end:]), axis=0)
        # Grid search over hyperparameters
        for max_depth, min_samples_split in product(hyperparameters['max_depth'], hyperparameters['min_samples_split']):
            dt_classifier = DecisionTree('mse', max_depth=max_depth, min_samples=min_samples_split)
            dt_classifier.fit(pd.DataFrame(training_set, columns=['f1', 'f2']), pd.Series(training_labels, name='Y'))
            val_pred = dt_classifier.predict(pd.DataFrame(val_set, columns=['f1', 'f2']))
            val_accuracy = accuracy(pd.Series(val_pred), pd.Series(val_labels))

            # Update optimal hyperparameters if better accuracy is found
            if val_accuracy > best_accuracy:
                best_max_depth = max_depth
                best_min_samples_split = min_samples_split
                best_accuracy = val_accuracy
        outer_fold_opt_depth.append(best_max_depth)       
        outer_fold_opt_minsamples.append(best_min_samples_split)
    # Train decision tree on combined train + validation data with optimal hyperparameters
    dt_classifier_optimal = DecisionTree('mse', max_depth=best_max_depth, min_samples=best_min_samples_split)
    dt_classifier_optimal.fit(pd.DataFrame(train_set, columns=['f1', 'f2']), pd.Series(train_labels, name='Y'))

    # Evaluate accuracy on test data
    test_pred = dt_classifier_optimal.predict(pd.DataFrame(test_set, columns=['f1', 'f2']))
    test_accuracy = accuracy(pd.Series(test_pred), pd.Series(test_labels))
    outer_fold_accuracies.append(test_accuracy)

print("Optimal Depth Decision Tree Accuracies, Optimal Depth and Minimum sample split for Each Outer Fold:")
for outer_fold, accuracy_value in enumerate(outer_fold_accuracies, 1):
    print(f"Outer Fold {outer_fold}: Accuracy = {accuracy_value:.4f}, optimal depth = {outer_fold_opt_depth[outer_fold]}, min sample split = {outer_fold_opt_minsamples[outer_fold]}")




