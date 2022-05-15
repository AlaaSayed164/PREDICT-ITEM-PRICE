import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score

######
#import matplotlib.pyplot as plt

####
train=pd.read_csv("train.csv")

train = train.replace(' ', np.nan)

train["Y"] = train["Y"].astype(str)

train["X9"].fillna( method ='ffill', inplace = True)
train["X2"].fillna( method ='ffill', inplace = True)
train.isnull().sum()

train["X3"]=train["X3"].replace(to_replace=["LF","low fat"],value="Low Fat")
train["X3"]=train["X3"].replace(to_replace=["reg"],value="Regular")

m=train["X4"].mean()
train["X4"]=train["X4"].replace(to_replace=0,value=m)

train.drop("X1",inplace=True,axis=1)
train.drop("X7",inplace=True,axis=1)

label=preprocessing.LabelEncoder()
train["X3"]=label.fit_transform(train["X3"])
train["X9"]=label.fit_transform(train["X9"])
train["X10"]=label.fit_transform(train["X10"])

#print (train["X5"].unique)
train.drop("X5",inplace=True,axis=1)

train.rename(columns={'X2': 'Weight of Item', 'X3': 'Amount of Fats in Item','X4': 'area allocated for item in store ','X6': 'Item Price','X8': 'Store Establishment Year','X9': 'Store Size','X10': 'Store Location Type','Y':'label'}, inplace=True)

x=train.corr()
train.drop("Item Price",inplace=True,axis=1)

x1=train[['Weight of Item','Amount of Fats in Item','area allocated for item in store ','Store Establishment Year','Store Size','Store Location Type']]
y=train['label']
x_train, x_test, y_train, y_test = train_test_split(x1, y,test_size = 0.1,random_state = 0)
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
KNN_pred =model.predict(x_test)
acc1=accuracy_score(y_test, KNN_pred)
print("knn: ",acc1)
#^^^^^^^^^^
from sklearn.ensemble import GradientBoostingClassifier
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(x_train, y_train)
GBC_pred = gb_clf2.predict(x_test)
acc2=accuracy_score(y_test, GBC_pred)
print("GradientBoostingClassifier: ",acc2)
#^^^^^^^^^^
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(x_train, y_train)
RFC_pred= RF.predict(x_test)
acc3=accuracy_score(y_test, RFC_pred)
print("RandomForestClassifier: ",acc3)
#^^^^^^^^^^
from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
model = GaussianNB()
# Train the model using the training sets
model.fit(x_train, y_train)
#Predict Output
GNB_pred =model.predict(x_test)
acc4=accuracy_score(y_test, GNB_pred)
print("gaussian: ",acc4)
#^^^^^^^^^^
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(x_train, y_train)
rbf_pred = rbf.predict(x_test)
acc5=accuracy_score(y_test, rbf_pred)
print("svm using rbf: ",acc5)

################################################################################
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
classfierAlgo = ['knn', 'GradientBoostingClassifier', 'RandomForestClassifier', 'gaussian', 'svm']
accurocy = [acc1,acc2,acc3,acc4,acc5]
ax.bar(classfierAlgo,accurocy)
plt.show()



#################################################
test=pd.read_csv("test.csv")


test = test.replace(' ', np.nan)



test["X9"].fillna( method ='ffill', inplace = True)
test["X2"].fillna( method ='ffill', inplace = True)
test.isnull().sum()

test["X3"]=test["X3"].replace(to_replace=["LF","low fat"],value="Low Fat")
test["X3"]=test["X3"].replace(to_replace=["reg"],value="Regular")

m=test["X4"].mean()
test["X4"]=test["X4"].replace(to_replace=0,value=m)

test.drop("X1",inplace=True,axis=1)
test.drop("X7",inplace=True,axis=1)

label=preprocessing.LabelEncoder()
test["X3"]=label.fit_transform(test["X3"])
test["X9"]=label.fit_transform(test["X9"])
test["X10"]=label.fit_transform(test["X10"])

#print (test["X5"].unique)
test.drop("X5",inplace=True,axis=1)

test.rename(columns={'X2': 'Weight of Item', 'X3': 'Amount of Fats in Item','X4': 'area allocated for item in store ','X6': 'Item Price','X8': 'Store Establishment Year','X9': 'Store Size','X10': 'Store Location Type','Y':'label'}, inplace=True)

x=test.corr()
test.drop("Item Price",inplace=True,axis=1)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x1, y)
y_predictKnn =model.predict(test)
#^^^^^^^^^^

gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(x1, y)
y_predictGb = gb_clf2.predict(test)
#^^^^^^^^^^

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(x1, y)
y_predictRFC=RF.predict(test)

#^^^^^^^^^^
model = GaussianNB()
model.fit(x1, y)
y_predictGNB =model.predict(test)

#^^^^^^^^^^
rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(x1, y)
y_predictSVM = rbf.predict(test)
