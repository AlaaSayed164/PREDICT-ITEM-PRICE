# PREDICT-ITEM-PRICE

This project was a Kaggle competition classification problem  in subject pattern recognition in faculty 
we have three datasets train, test,sample_submission 


It was 2 milestones  :

The first milestone Preprocessing technique "There is a pdf with code explanation ": 

1-first we check if any column contains NAN value and deal with it.

2-check for the unique value of a column for example ( "LF" ,"low fat" ,and "Low FAT" is the same meaning ).

3-check for zero value.

4-check for the range between the number and outlier then use the minmax scaler.

5-Analysis On Dataset with apply correlation matrix to show the correlation between a feature and label then drop a column that has small relation with my label.

6-convert all object (string) to number We use one hot encoder(to avoid large number and gives priority) and a label encoder (to avoid a large number of feature).

7-Split dataset into train and test. then use  Regression techniques like linear model ridge and -Neural Network MLPRegressor.


The second milestone uses Machine learning models "There is a pdf with code explanation ": 

1-Split previous dataset into train and test

2-use classification techniques such as K Neighbors Classifier, Gradient boosting classifiers, A random forest, Gaussian Naive Bayes, SVM Classifier, and RBF Kernel

3- show accuracy score of our techniques in run time



