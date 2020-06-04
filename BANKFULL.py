# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:15:03 2020

@author: intel
"""

#importing neccesary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from ml_metrics import rmse
from sklearn.linear_model import LogisticRegression
#importing the csv dataset using pd.read function
bankfull=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\ASSIGNMENT6//bankfull.csv")
bankfull.columns
#columns names
#(['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       #'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
      # 'previous', 'poutcome', 'y']
#'y' is my output varauble to predict
#label encoder() using for levels of categrical features into numerical values
Le=preprocessing.LabelEncoder()
bankfull["Job "]=Le.fit_transform(bankfull["job"])
bankfull["Marital "]=Le.fit_transform(bankfull["marital"])
bankfull["Education "]=Le.fit_transform(bankfull["education"])
bankfull["Default "]=Le.fit_transform(bankfull["default"])
bankfull["Housing "]=Le.fit_transform(bankfull["housing"])
bankfull["Contact "]=Le.fit_transform(bankfull["contact"])
bankfull["Month "]=Le.fit_transform(bankfull["month"])
bankfull["Poutcome "]=Le.fit_transform(bankfull["poutcome"])
bankfull["Loan "]=Le.fit_transform(bankfull["loan"])
bankfull["Y "]=Le.fit_transform(bankfull["y"])
#Below i'm removing the unwanted variables from datasets using drop()
bankfull = bankfull.drop('job',axis = 1)
bankfull = bankfull.drop('marital',axis = 1)
bankfull = bankfull.drop('education',axis = 1)
bankfull = bankfull.drop('default',axis = 1)
bankfull = bankfull.drop('housing',axis = 1)
bankfull = bankfull.drop('contact',axis = 1)
bankfull = bankfull.drop('poutcome',axis = 1)
bankfull = bankfull.drop('month',axis = 1)
bankfull = bankfull.drop('loan',axis = 1)
bankfull = bankfull.drop('y',axis = 1)
#Finding mean,median,mode,sd,variance,etc..
bankfull.describe()
             #age        balance  ...         pdays      previous
count  45211.000000   45211.000000  ...  45211.000000  45211.000000
mean      40.936210    1362.272058  ...     40.197828      0.580323
std       10.618762    3044.765829  ...    100.128746      2.303441
min       18.000000   -8019.000000  ...     -1.000000      0.000000
25%       33.000000      72.000000  ...     -1.000000      0.000000
50%       39.000000     448.000000  ...     -1.000000      0.000000
75%       48.000000    1428.000000  ...     -1.000000      0.000000
max       95.000000  102127.000000  ...    871.000000    275.000000


#first i have to rename my converted varaibles  for avoiding the "AttributeError"
bankfull = bankfull.rename(columns={'Job ': 'job'})
bankfull = bankfull.rename(columns={'Marital ': 'marital'})
bankfull = bankfull.rename(columns={'Education ': 'education'})
bankfull = bankfull.rename(columns={'Default ': 'default'})
bankfull = bankfull.rename(columns={'Housing ': 'housing'})
bankfull = bankfull.rename(columns={'Contact ': 'contact'})
bankfull = bankfull.rename(columns={'Month ': 'month'})
bankfull = bankfull.rename(columns={'Poutcome ': 'poutcome'})
bankfull = bankfull.rename(columns={'Loan ': 'loan'})
bankfull = bankfull.rename(columns={'Y ': 'y'})
#Next i'm going for the EDA section using different visualization
sb.countplot(x="job",data=bankfull)
sb.countplot(x="marital",data=bankfull)
sb.countplot(x="education",data=bankfull)
sb.countplot(x="housing",data=bankfull)
sb.countplot(x="contact",data=bankfull)
sb.countplot(x="month",data=bankfull)
sb.countplot("poutcome",data=bankfull)
sb.countplot("loan",data=bankfull)
sb.countplot("y",data=bankfull)
#next i would like to see the bar visualization of two variables using crosstab function.
pd.crosstab(bankfull.y,bankfull.age).plot(kind="bar")
pd.crosstab(bankfull.marital,bankfull.age).plot(kind="bar")
pd.crosstab(bankfull.marital,bankfull.job).plot(kind="bar")
pd.crosstab(bankfull.marital,bankfull.y).plot(kind="bar")
pd.crosstab(bankfull.y,bankfull.job).plot(kind="bar")
pd.crosstab(bankfull.y,bankfull.education).plot(kind="bar")
pd.crosstab(bankfull.y,bankfull.default).plot(kind="bar")
plt.hist(bankfull.job)
plt.hist(bankfull.y)
#Next i'm going to check the null values from the data
bankfull.isnull().sum()
#age          0
#balance      0
#day          0
#duration     0
#campaign     0
#pdays        0
#previous     0
#job          0
#marital      0
#education    0
#default      0
#housing      0
#contact      0
#month        0
#poutcome     0
#loan         0
#y            0
#Above i can see there is no null values inside my data
#Next I'm going to build my model
#'y' is my output variable and rest are my input variable
X=bankfull.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]#here i'm creating a new dataset called 'X' with input variables
bankfull.columns[16]
#'y'
Y=bankfull.iloc[:,[16]]#here i'm creating a new variable called 'Y' with output variable 'y'
Classifier=LogisticRegression()
Classifier.fit(X,Y)#fitting my variables called 'X','Y'
#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                  # intercept_scaling=1, l1_ratio=None, max_iter=100,
                  # multi_class='warn', n_jobs=None, penalty='l2',
                  # random_state=None, solver='warn', tol=0.0001, verbose=0,
                  # warm_start=False)
#Next i would like to get the coefficents                   
Classifier.coef_
#array([[ 6.27549547e-03,  1.76625369e-05, -5.73933976e-03,
       #  3.94392799e-03, -1.29913301e-01,  3.21334543e-03,
       # 8.38700459e-02,  7.57412614e-03,  1.83638529e-01,
       #  1.98477644e-01, -3.25614290e-01, -1.07771039e+00,
       # -6.43742664e-01,  3.87084845e-02,  1.67625694e-01,
       # -7.18124008e-01]])
#to get the possibilities of my input variable called 'X'
ypred=Classifier.predict(X)
ypred
#array([0, 0, 0, ..., 1, 0, 1])
#adding the predicted possibilities to the dataset called 'ypred'
bankfull["ypred"]=ypred
yprob=pd.DataFrame(Classifier.predict_proba(X.iloc[:,:]))
#Above dataframing the predicted probabilities of 'X' and store it in a new variable called 'yprob'
#Next I'm taking the input strings and joins them into one using concat() function and store it in a new variable called 'new_df'
new_df=pd.concat([bankfull,yprob],axis=1)
#cut off value is '0.5'

#Next i would like to import the confusion matrix to see the true vales inside my datasets
from sklearn.metrics import confusion_matrix
#Next i would like to see the table of my actual amd predicted values using confusion matrix
confusion_matrix=confusion_matrix(Y,ypred)
confusion_matrix
#([[39130,   792],
# [ 4142,  1147]],
#I need to calculate my accuracy from this table and store it in a new variable called accuracy
Accuracy=(39130+1147)/45211
Accuracy
#0.890867266815598

#Next i would likr to see the ROC plot for the given dataset
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr,tpr,threshold = roc_curve(Y,ypred)
fpr
#array([0.        , 0.01983869, 1.        ])
tpr
#array([0.        , 0.21686519, 1.        ])
threshold
#array([2, 1, 0])
#next i would to calculate the auc score
auc=roc_auc_score(Y,ypred)
auc
# 0.5985132532355658
#to see the ROC curve
import matplotlib.pyplot as plt
plt.plot(fpr,tpr,color="red",label="ROC")
#