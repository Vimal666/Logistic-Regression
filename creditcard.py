# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:48:14 2020

@author: Vimal PM
"""
#importing neccesary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing
from ml_metrics import rmse
from sklearn.linear_model import LogisticRegression

#importing the datasets using pd.read function
creditcard=pd.read_csv("D:\DATA SCIENCE\ASSIGNMENT\ASSIGNMENT6\creditcard.csv")
#label encoder() using for levels of categrical features into numerical values
Le=preprocessing.LabelEncoder()
creditcard['Card']=Le.fit_transform(creditcard['card'])
creditcard['Owner']=Le.fit_transform(creditcard['owner'])
creditcard['Selfemp']=Le.fit_transform(creditcard['selfemp'])

#Below i'm removing the unwanted variables from datasets using drop()
creditcard = creditcard.drop('card',axis = 1)
creditcard = creditcard.drop('owner',axis = 1)
creditcard = creditcard.drop('selfemp',axis = 1)
creditcard = creditcard.drop('Unnamed: 0',axis = 1)
#columns names
creditcard.columns
#Index(['reports', 'age', 'income', 'share', 'expenditure', 'dependents',
  # 'months', 'majorcards', 'active', 'Card', 'Owner', 'Selfemp'],
     


#Next i would like to calculate my mean,medain,mode,varience,standard error, etc from the dataset...
creditcard.describe()
          #reports          age  ...        Owner      Selfemp
count  1319.000000  1319.000000  ...  1319.000000  1319.000000
mean      0.456406    33.213103  ...     0.440485     0.068992
std       1.345267    10.142783  ...     0.496634     0.253536
min       0.000000     0.166667  ...     0.000000     0.000000
25%       0.000000    25.416670  ...     0.000000     0.000000
50%       0.000000    31.250000  ...     0.000000     0.000000
75%       0.000000    39.416670  ...     1.000000     0.000000
max      14.000000    83.500000  ...     1.000000     1.000000
#Next i'm going for my EDA section using different visualizations
sb.countplot(x="Card",data=creditcard)#here i can say the people who accepted the card is more than the people who's not accepted the card
pd.crosstab(creditcard.Card,creditcard.Owner).plot(kind="bar")#Here i can the person who's not accepted the card with person's  who's not owns the house is very high compare to persons who's owns the house. AND also in the person who's accepted the card with person's  who's not owns the house is very high compare to persons who's owns the house
#next i'm going to see the income countplot
sb.countplot(x="income",data=creditcard)
pd.crosstab(creditcard.income,creditcard.Selfemp).plot(kind="bar")
#next i would like to see the crosstab function of majorcards with selfemployment and see it as barplot
pd.crosstab(creditcard.majorcards,creditcard.Selfemp).plot(kind="bar")
#crosstab function of active with Owner and see it as barplot
pd.crosstab(creditcard.active,creditcard.Owner).plot(kind="bar")
#crosstab function of active with Owner and see it as barplot
pd.crosstab(creditcard.reports,creditcard.months).plot(kind="bar")
#boxlpot visualizations
sb.boxplot(data = creditcard,orient = "v")
sb.boxplot(x="Card",y="Owner",data=creditcard,palette = "hls")
sb.boxplot(x="income",y="Selfemp",data=creditcard,palette = "hls")
sb.boxplot(x="active",y="Owner",data=creditcard,palette = "hls")
sb.boxplot(x="Card",y="majorcards",data=creditcard,palette = "hls")
sb.boxplot(x="share",y="dependents",data=creditcard,palette = "hls")
sb.boxplot(x="Card",y="Owner",data=creditcard,palette = "hls")
sb.boxplot(x="months",y="reports",data=creditcard,palette = "hls")

#To get the null values from the datasets
creditcard.isnull().sum()
#reports        0
#age            0
#income         0
#share          0
#expenditure    0
#dependents     0
#months         0
#majorcards     0
#active         0
#Card           0
#Owner          0
#Selfemp        0
#from above analyis I can say there is no null values in my dataset
#So directly going for model building
x=creditcard.iloc[:,[0,1,2,3,4,5,6,7,8,10,11]]#here i'm  adding my inputs variables in a new variable called 'x'
creditcard.columns[9]
#'Card'
y=creditcard.iloc[:,[9]]#here i'm adding my ouput variable and store it in a new variable called 'y'

classifier=LogisticRegression()
classifier.fit(x,y)#fitting the x and y variables
#getting the coefficents
classifier.coef_
[[-1.65509892e+00, -3.11259045e-03, -2.05627500e-01,
        -6.17009980e-04,  1.61855364e+00, -6.64242088e-01,
        -1.60414471e-03,  3.43439831e-02,  7.65660467e-02,
         6.20454990e-01,  2.29555725e-01]])
classifier.predict_proba(x)#here i'm getting the probabilities of my input variables
([[0.00000000e+00, 1.00000000e+00],
       [2.01651286e-06, 9.99997983e-01],
       [1.49800861e-09, 9.99999999e-01],
       ...,
       [0.00000000e+00, 1.00000000e+00],
       [0.00000000e+00, 1.00000000e+00],
       [0.00000000e+00, 1.00000000e+00]])
ypred=classifier.predict(x)#here i'm predicting the possibilities of 'x'
ypred
#adding the predicted variable to the datasets
creditcard["ypred"]=ypred
#dataframing the predicted probabilities of 'x' and store it in a new variable called 'yprob'
yprob=pd.DataFrame(classifier.predict_proba(x.iloc[:,:]))
#next i'm joining my inputs strings into one using concat function and store it in a new variable called 'new_df'
new_df=pd.concat([creditcard,yprob],axis=1)
from sklearn.metrics import confusion_matrix
#confusion matrix table is using for to see the true values
confusion_matrix = confusion_matrix(y,ypred)#here i would like to see the confusin matrix table of 'y' variable and y'ypred'
confusion_matrix
([[ 295,    1],
  [  23, 1000]],
#Accuracy=295+1000/1319
#Acuracy=0.981

#Next i would like to caluculate my aoc score and tpr,fpr,threshold values
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr,tpr,threshold=roc_curve(y,ypred)
fpr
#array([0.        , 0.00337838, 1.        ])
tpr
#array([0.        , 0.97751711, 1.        ])
threshold
#array([2, 1, 0])
#Next I'm going ro get my auc score
auc=roc_auc_score(y,ypred)
auc
#0.9870693640854931

#To see the roc curve
import matplotlib.pyplot  as plt
plt.plot(fpr,tpr,color="red",label="ROC")
