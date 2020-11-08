import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV

data = './heart.csv'
df = pd.read_csv(data)
print(df.shape)
print(df.head())

print(df.isnull().sum())

print(df.columns)
X=df.drop('target',axis=1)
y = df.target
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size = 0.10, random_state = 10)

print('Training Set :',len(X_train))
print('Test Set :',len(X_test))
print('Training labels :',len(y_train))
print('Test Labels :',len(y_test))

param ={
            'n_estimators': [100, 500, 1000,1500, 2000],
    'max_depth' :[2,3,4,5,6,7],
           
        }
m2 = GridSearchCV(RandomForestClassifier(), grid,cv=5)
m2.fit(X_train, y_train)
print(m2.best_params_)
pred2 = m2.predict(X_test)
print(classification_report(y_test, pred2))

grid ={
            'n_estimators': [100, 500, 1000,1500, 2000],
            'max_depth' :[2,3,4,5,6,7],
    	    'learning_rate': [0.01,0.1,0.01]
           
        }

m3 = GridSearchCV(GradientBoostingClassifier(), grid, cv=5)
m3.fit(X_train, y_train) 
print(m3.best_params_)
pred3 = m3.predict(X_test)
print(classification_report(y_test, pred3))