# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Seaborn is used to display statistical data using various graphs
import seaborn as sns

from collections import Counter
#Counter is used to count the number of items 


sns.set(style='white', context='notebook', palette='deep')

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

def detect_outliers(df,n,features):
    """
    Tukey method.
    http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < (Q1 - outlier_step)  ) | (df[col] > (Q3 + outlier_step) )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

#Display the outliers here 
train.loc[Outliers_to_drop]
#Drop the outliers 
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

# Infos of null and missing values
train.info()
train.isnull().sum()

### Summarize data
# Summarie and statistics
train.describe()


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(train[["Survived","SibSp","Parch","Age","Fare"]].corr(),
                annot=True, fmt = ".2f", cmap = "coolwarm")


# Explore SibSp feature vs Survived where SibSp is number of sibings or spouses
g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , 
                   palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
train[["SibSp","Survived"]].groupby('SibSp').mean()

# Explore Parch feature vs Survived
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
train[["Parch","Survived"]].groupby('Parch').mean()

# Explore Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
# Explore Age distibution 
g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g = g.legend(["Not Survived","Survived"])

#To check how many fare's are missing 
train["Fare"].isnull().sum()
test["Fare"].isnull().sum()
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# Apply log to Fare to reduce skewness distribution
train["Fare"] = train["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
test["Fare"] = test["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(train["Fare"], color="b", label="Skewness : %.2f"%(train["Fare"].skew()))
g = g.legend(loc="best")

#Sex vs Survival 
g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")
train[["Sex","Survived"]].groupby('Sex').mean()


train.isnull().sum()
test.isnull().sum()
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())


train['Embarked'].loc[train['Embarked'] =='S'] = 1
train['Embarked'].loc[train['Embarked'] == 'C'] = 2
train['Embarked'].loc[train['Embarked'] == 'Q'] = 3
test['Embarked'].loc[test['Embarked'] =='S'] = 1
test['Embarked'].loc[test['Embarked'] == 'C'] = 2
test['Embarked'].loc[test['Embarked'] == 'Q'] = 3

y = train.iloc[:, [1]].values
train.isnull().sum()
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].median())
test.isnull().sum()
train = train.iloc[:, [2,4,5,6,7,9,11]].values
test = test.iloc[:, [1,3,4,5,6,8,10]].values

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
train[:,1] = labelencoder_x.fit_transform(train[:,1])
test[:,1] = labelencoder_x.fit_transform(test[:,1])


'''


#We need to dummy enocde it with each column for each country
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
train = onehotencoder.fit_transform(train).toarray()
test = onehotencoder.fit_transform(test).toarray()

onehotencoder = OneHotEncoder(categorical_features = [5])
train = onehotencoder.fit_transform(train).toarray()
test = onehotencoder.fit_transform(test).toarray()

train = train[:, [1,2,3,4,5,6,7,8,9,10,11,12,13]]
test = test[:, [1,2,3,4,5,6,7,8,9,10,11,12,13]]
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(train)
test = sc.transform(test)

from sklearn.svm import SVC
classifier1 = SVC(kernel = 'rbf', probability=True)
classifier1.fit(train, y)
print("\n SVC : ",classifier1.score(train,y))
y_pred1 = classifier1.predict(test)

from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression()
classifier2.fit(train, y)
print("\n Logistic Regression: ",classifier2.score(train,y))
y_pred2 = classifier2.predict(test)


from sklearn.neighbors import KNeighborsClassifier
classifier3 = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2,
                                   algorithm = 'auto')
classifier3.fit(train, y)
print("\n KNN : ",classifier3.score(train,y))
y_pred3 = classifier3.predict(test)

from sklearn.naive_bayes import GaussianNB
classifier4 = GaussianNB()
classifier4.fit(train, y)
print("\n Naive Bayes: ",classifier4.score(train,y))
y_pred4 = classifier4.predict(test)

from sklearn.ensemble import RandomForestClassifier
classifier5 = RandomForestClassifier(n_estimators = 50, criterion = 'gini')
classifier5.fit(train, y)
print("\n Random Forest: ",classifier5.score(train,y))
y_pred5 = classifier5.predict(test)

from sklearn.ensemble import GradientBoostingClassifier
classifier6 = GradientBoostingClassifier(n_estimators=100)
classifier6.fit(train,y)
y_pred6 = classifier6.predict(test)
print("\n Gradient Boosting: ",classifier6.score(train,y))

from sklearn.tree import DecisionTreeClassifier
classifier7 = DecisionTreeClassifier(criterion = 'entropy')
classifier7.fit(train, y)
y_pred7 = classifier7.predict(test)
print("\n Decision Tree: ",classifier7.score(train,y))

from sklearn.ensemble import AdaBoostClassifier
classifier8 = AdaBoostClassifier(algorithm = 'SAMME', n_estimators = 50)
classifier8.fit(train,y)
y_pred8 = classifier8.predict(test)
print("\n ADA boost: ",classifier7.score(train,y))


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
y_pred = (y_pred1 + y_pred3 + y_pred6 + y_pred8 + y_pred7)/5
for z in range(0,len(y_pred)): 
    if y_pred[z] > 0.5:
        y_pred[z] = 1
    else:
        y_pred[z] = 0
dataset3 = pd.read_csv('gender_submission.csv')
dataset3['Survived'] = y_pred



