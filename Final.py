# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:03:32 2018

@author: reach
"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

#importing dataframes
Train=pd.read_csv('train.csv')
Test=pd.read_csv('test.csv')

#checking survival rate per feature
Train[['Pclass','Survived']].groupby('Pclass').mean()
Train[['Sex','Survived']].groupby('Sex').mean()
Train[['SibSp','Survived']].groupby('SibSp').mean()
Train[['Parch','Survived']].groupby('Parch').mean()

#filling missing values
Train=Train.fillna(Train.mean())
Test=Test.fillna(Train.mean())

#FacetGrid
g=sb.FacetGrid(Train,col='Survived')
g.map(plt.hist,'Age',bins=10)
g.map(plt.hist,'Fare',bins=10)


f=sb.FacetGrid(Train,col='Survived')
f.map(plt.hist,'Fare',bins=10)

#removing unimportant features
Train=Train.drop(['PassengerId'],axis=1)
Train=Train.drop(['Cabin','Ticket'],axis=1)
Test=Test.drop(['Cabin','Ticket'],axis=1)

#adding new feature family size in both datasets
Train['FamilySize']=Train['Parch']+Train['SibSp']+1
Train=Train.drop(['Parch','SibSp'],axis=1)
Test['FamilySize']=Test['Parch']+Test['SibSp']+1
Test=Test.drop(['Parch','SibSp'],axis=1)

#categorical mapping
Train['Sex'] = Train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
Test['Sex'] = Test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

Train=Train.dropna()
Train['Embarked'] = Train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
Test['Embarked'] = Test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



#Categorising titles
combine=[Train,Test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(Train['Title'], Train['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

Train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
Train=Train.drop(['Name'],axis=1)
Test=Test.drop(['Name'],axis=1)

#creating training set
X_train=Train.iloc[:,1:].values
Y_train=Train.iloc[:,[0]].values
#creating test set
X_test=Test.iloc[:,1:].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_Train=sc.fit_transform(X_train)
X_Test=sc.transform(X_test)



#using random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=100)
classifier.fit(X_train,Y_train)

Y_pred=classifier.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": Test["PassengerId"],
        "Survived": Y_pred
    }, )
    

submission.to_csv(path_or_buf='submission.csv', index=False)