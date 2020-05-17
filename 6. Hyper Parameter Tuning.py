# -*- coding: utf-8 -*-
import pandas as pd 
import string
import re 
import numpy as np 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.tokenize import word_tokenize,sent_tokenize 
from nltk.stem import PorterStemmer, WordNetLemmatizer
import unidecode
import pickle
from sklearn import datasets,metrics,cross_validation,linear_model
import sklearn.ensemble as ske
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support
from pprint import pprint
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

############################Score function 
def score(model,x_test,y_test):
    y_test_f=y_test.values.ravel()
    scores_f=pd.DataFrame(model.predict_proba(x_test)) 
    scores_f["Validation"]=y_test
    scores_f=scores_f.sort_values(1,ascending=False)  
    scores_f=scores_f.reset_index()
    del scores_f["index"]
    a=len(scores_f)
    a=int(0.3*a+1)
    scores_c=scores_f.iloc[0:a,:]
    m=pd.DataFrame(scores_f['Validation']==1).sum()
    n=pd.DataFrame(scores_c['Validation']==1).sum()
    recall_f = n[0]/m[0]
    precision_f = n[0]/a
    score_f={"recall":recall_f,"precision":precision_f}
    pprint(score_f)
    return score_f

##################################Using values nearby best parameter acheived from RandomizedSearchCV ,calculated ROC_AUC area
y_train_array=y_train.values.ravel()
output=pd.DataFrame(columns=['max_features','min_samples_leaf','min_samples_split','n_estimators',"recall","precision"])
prob_score={}
for max_features in [40,45,50]: 
    for min_samples_leaf in [1]:
        for min_samples_split in [2]:
            for n_estimators in [200,300,500,700,900,1100,1300,1600,2000,3000]:
                model= ske.RandomForestClassifier(bootstrap=True, criterion='entropy',
                                    max_features=max_features,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, n_jobs=1, random_state=None)
                model.fit(x_train,y_train_array)
                score1=score(model,x_test,y_test)
                output=output.append({
                                  'max_features':max_features,
                                  'min_samples_leaf':min_samples_leaf,
                                  'min_samples_split':min_samples_split,
                                  'n_estimators':n_estimators,
                                  'recall':score1["recall"],
                                  'precision':score1["precision"]}, ignore_index=True)
                    
                prob = list(pd.DataFrame(model.predict_proba(x_test)))
                ty="%s_%s_%s_%s"%(max_features,min_samples_leaf,min_samples_split,n_estimators)
                prob_score[ty]=prob    

prob_score=pd.DataFrame(prob_score)
prob_score.to_csv(".csv",index=False)
output.to_csv("output_3.csv",index=False)
output_f=output[[ 'max_features', 'min_samples_leaf', 'min_samples_split',
       'n_estimators','roc_auc_area']]

#extracting maximum roc-auc area 
maximum=output_f["roc_auc_area"].max()
row=output_f[output_f["roc_auc_area"]==maximum]
print(row)

y_test["Validation"].value_counts()














