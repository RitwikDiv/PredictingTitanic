

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')

dataset['Family'] =  dataset["Parch"] + dataset["SibSp"]
dataset['Family'].loc[dataset['Family'] > 0] = 1
dataset['Family'].loc[dataset['Family'] == 0] = 0
dataset = dataset.drop(['SibSp','Parch'], axis=1)

dataset = dataset.iloc[:, [1,2,4,5,9,10]]

dataset['Age'] = dataset.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
dataset.loc[(dataset['Age'] > 16), 'Age'] = 1

dataset = dataset.fillna(method = 'ffill')
x = dataset.iloc[:, [1,2,3,4,5]].values
y = dataset.iloc[:, [0]].values

dataset2 = pd.read_csv('test.csv')

dataset2['Family'] =  dataset2["Parch"] + dataset2["SibSp"]
dataset2['Family'].loc[dataset2['Family'] > 0] = 1
dataset2['Family'].loc[dataset2['Family'] == 0] = 0
dataset2 = dataset2.drop(['SibSp','Parch'], axis=1)
dataset2['Age'] = dataset2.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))
dataset2 = dataset2.iloc[:, [1,3,4,8,9]]
dataset2.loc[ dataset2['Age'] <= 16, 'Age'] = 0
dataset2.loc[(dataset2['Age'] > 16), 'Age'] = 1
dataset2 = dataset2.fillna(method = 'ffill')
x_test = dataset2.iloc[:,].values

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
x[:,1] = labelencoder_x.fit_transform(x[:,1])
x_test[:,1] = labelencoder_x.fit_transform(x_test[:,1])
x[:,3] = labelencoder_x.fit_transform(x[:,3])
x_test[:,3] = labelencoder_x.fit_transform(x_test[:,3])



#We need to dummy enocde it with each column for each country
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
x_test = onehotencoder.fit_transform(x_test).toarray()

onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()
x_test = onehotencoder.fit_transform(x_test).toarray()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
x_test = sc.transform(x_test)


from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', probability=True)
classifier1.fit(x, y)
print("\n SVC : ",classifier1.score(x,y))
y_pred1 = classifier1.predict(x_test)

from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression()
classifier2.fit(x, y)
print("\n Logistic Regression: ",classifier2.score(x,y))
y_pred2 = classifier2.predict(x_test)


from sklearn.neighbors import KNeighborsClassifier
classifier3 = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2,
                                   algorithm = 'auto')
classifier3.fit(x, y)
print("\n KNN : ",classifier3.score(x,y))
y_pred3 = classifier3.predict(x_test)

from sklearn.naive_bayes import GaussianNB
classifier4 = GaussianNB()
classifier4.fit(x, y)
print("\n Naive Bayes: ",classifier4.score(x,y))
y_pred4 = classifier4.predict(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier5 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier5.fit(x, y)
print("\n Random Forest: ",classifier5.score(x,y))
y_pred5 = classifier5.predict(x_test)

from sklearn.ensemble import GradientBoostingClassifier
classifier6 = GradientBoostingClassifier(n_estimators=100)
classifier6.fit(x,y)
y_pred6 = classifier6.predict(x_test)
print("\n Gradient Boosting: ",classifier6.score(x,y))

from sklearn.tree import DecisionTreeClassifier
classifier7 = DecisionTreeClassifier(criterion = 'entropy')
classifier7.fit(x, y)
y_pred7 = classifier7.predict(x_test)
print("\n Decision Tree: ",classifier7.score(x,y))

from mlxtend.classifier import StackingClassifier
classifier8 = StackingClassifier(classifiers =[classifier1,classifier3,classifier4,
                                   classifier5,classifier6, classifier7], meta_classifier=classifier2)
classifier8.fit(x, y)
print("\n Ensemble Vote Classifier: ",classifier8.score(x,y))
y_pred8 = classifier8.predict(x_test)

'''
from sklearn import model_selection

for clf, label in zip([classifier1,classifier3,classifier4,
                                    classifier5,classifier6, classifier7,
                                    classifier8],
                       ['SVC', 
                       'KNN', 
                       'Naive Bayes',
                       'Random Forests',
                       'Gradient Boosting',
                       'Decision Tree',
                       'Stacking Vote Classifier'
                       ]):

    scores = model_selection.cross_val_score(clf, x, y, 
                                              cv=2, scoring='accuracy')
'''


# Predicting the Test set results
y_pred = (y_pred1 + y_pred3 + y_pred4 + y_pred5 +y_pred6+ y_pred7 +y_pred8)/7
for z in range(0,len(y_pred)): 
    if y_pred[z] > 0.4:
        y_pred[z] = 1
    else:
        y_pred[z] = 0
dataset3 = pd.read_csv('gender_submission.csv')
dataset3['Survived'] = y_pred


'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
'''