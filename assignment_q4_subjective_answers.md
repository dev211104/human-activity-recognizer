# Time Complexity Analysis

## Train time analysis in our implementation

#### Training time is plotted against N*log(N)*M 

1) **Real Input Discrete Output**
![Real input Discrete Output Train time](plots/Ri_Do_train.png)

2) **Real Input Real Output**
![Real input Real Output Train time](plots/Ri_Ro_train.png)

3) **Discrete Input Discrete Output**
![Discrete Input Discrete Output Train time](plots/Di_Do_train.png)

4) **Discrete Input Real Output**
![Discrete Input Real Output Train time](plots/Di_Ro_train.png)

## Theoretical time complexity for training the data.
#### O(N*log(N)*M)

##### The plot of train time agains N*log(N)*M is linear. 
##### It is more clear in the regression cases, with some small deviations in the classification cases.
##### Therefore Time complexity of our model is O(N*log(N)*M) which is same as therotical time complexity

--------------------------------------------

## Test time analysis in our implementation

#### Test time is plotted against N*log(N)*M 

1) **Real Input Discrete Output**
![Real input Discrete Output Train time](plots/Ri_Do_test.png)

2) **Real Input Real Output**
![Real input Real Output Train time](plots/Ri_Ro_test.png)

3) **Discrete Input Discrete Output**
![Discrete Input Discrete Output Train time](plots/Di_Do_test.png)

4) **Discrete Input Real Output**
![Discrete Input Real Output Train time](plots/Di_Ro_test.png)

## Theoretical time complexity for predicting the data.

#### O(depth)
##### Time complexity depends on the depth of the tree. While experimenting the time to predict data, the depth of tree remained constant.

##### The plot of predict time agains N*log(N)*M is constant with some small deviations.
##### Therefore Time complexity of our model is O(depth) which is same as therotical time complexity.

