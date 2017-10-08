# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 21:06:59 2017

@author: Abner Bian
"""

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn import ensemble, metrics 
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
import matplotlib  as mat
#mat.use('Agg')
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
hdd=pd.read_csv('C:/Users/admin/preprocession_data.csv')
hdd = hdd.drop(['Unnamed: 0','serial_number'], axis=1)
for i in range(0,hdd.shape[1]):
    hdd_mean=hdd.iloc[:,i].mean()
    hdd.iloc[:,i]=hdd.iloc[:,i].fillna(hdd_mean)
hdd_labels=hdd['failure']

hdd=hdd.drop('failure',axis=1)

#    hdd = SelectKBest(chi2, k=9).fit_transform(hdd,hdd_labels)
#    
#    
#   
for i in range(300):
    #hdd = SelectKBest(chi2, k=9).fit_transform(hdd,hdd_labels) 
    X_train, X_test, y_train, y_test = train_test_split(hdd, hdd_labels,test_size=0.2)
    smote = SMOTE(kind = "regular")
    X_train,y_train = smote.fit_sample(X_train, y_train)
    #clf=ensemble.RandomForestClassifier()
    clf =tree.DecisionTreeClassifier(max_depth= None, criterion= 'gini', min_samples_split= 3,min_samples_leaf= 2, max_leaf_nodes= 5)
    clf = clf.fit(X_train,y_train)
    preds=clf.predict_proba(X_test)
    preds_=clf.predict(X_test) 
    
    roc_auc=metrics.roc_auc_score(y_true=y_test, y_score=preds[:,1])
    print('roc_auc', roc_auc)
    print('NACC',metrics.recall_score(y_true=y_test,y_pred=preds_))
    print('accuracy',metrics.accuracy_score(y_true=y_test,y_pred=preds_))
    if((metrics.recall_score(y_true=y_test,y_pred=preds_)>0.8)&(metrics.accuracy_score(y_true=y_test,y_pred=preds_)>0.8)):
        break
#recall.append(metrics.recall_score(y_true=test_label,y_pred=preds_))
#accuracy.append(metrics.accuracy_score(y_true=test_label,y_pred=preds_))
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds[:,1])

fig = plt.figure()
plt.title('ROC curve')
plt.plot(fpr, tpr, 'b')

plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0,1.1])
plt.ylim([-0,1.1])
plt.ylabel('tpr')
plt.xlabel('fpr')
plt.show()
fig.savefig('ROC curve.png')       


























