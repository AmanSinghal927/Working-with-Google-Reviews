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
import enchant
from enchant.checker import SpellChecker
from fuzzywuzzy import fuzz
from sklearn import datasets,metrics,cross_validation,linear_model
import sklearn.ensemble as ske
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support
from pprint import pprint
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

raw_data=pd.read_csv(r"",encoding='latin-1')
raw_data=raw_data[raw_data["Comments_concat"]!="nan"]
raw_data=raw_data[raw_data["Comments_concat"]!=""]
raw_data=raw_data.drop_duplicates(subset="",keep="first",inplace=False)
raw_data= raw_data.apply(lambda x: x.astype(str).str.lower())

raw_data["Comments_concat"]=raw_data['Comments_concat'].str.replace('[{}]'.format(string.punctuation), '')

raw_data=raw_data.reset_index()
del raw_data["index"]

data=raw_data.copy()
#stopwords removal 
m = []
for i in range(len(data)):
    x = " ".join(re.split("[^\w\s]",data.loc[i,"Comments_concat"])) ## to be changed
    tokens = x.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    c = ' '.join(words)
    m.append(c)
data['Comments_concat'] = pd.DataFrame(m)

#removing words with less than 3 characters and removing accents from words in the concatenated comments ex. malaga 
Comments_concat_list=[]
for i in range(len(data)):
    Comments_concat_list.append(data.loc[i,"Comments_concat"])
for i in range(len(Comments_concat_list)):
    Comments_concat_list[i] = ' '.join([w for w in Comments_concat_list[i].split() if len(w)>2])
    accented_string=Comments_concat_list[i]
    unaccented_string = unidecode.unidecode(accented_string)
    Comments_concat_list[i]=str(unaccented_string)
for i in range(len(data)):
    data.loc[i,"Comments_concat"]=Comments_concat_list[i]

#for removing the letters like !@#ð$
for i in range(len(data)):
     data.loc[i,"Comments_concat"]=re.sub('[!@#ð$]', '', data.loc[i,"Comments_concat"])

data=data[data.loc[:,"Comments_concat"]!="nan"]
data=data.fillna("")
data = data[data['Comments_concat']!='']
data=data.reset_index()
del data['index']

data.to_csv("data_after_cleaning.csv")

##spelling correction 
d = enchant.DictWithPWL("en", "") ###reading bag_of_words.txt
chkr = SpellChecker(d)

for i in range(len(data)):
    print(i)
    chkr.set_text(data["Comments_concat"][i])
    for err in chkr:
        for j in range(len(d.suggest(err.word))):
            for k in range(len(lst)):
                if(fuzz.ratio(err.word,lst[k])>85):
                    data['Comments_concat'][i] = data["Comments_concat"][i].replace(err.word,lst[k])
                    print("found: ", err.word, "\t",lst[k], "\t",fuzz.ratio(err.word,lst[k]))
                    
data.to_csv("data_after_enchant.csv",encoding='latin-1')

#final data after enchant and appending it with already enchanted data
data=pd.read_csv(".csv",encoding='latin-1')
data=data[data["Comments_concat"]!="nan"]
data=data.drop_duplicates(subset="",keep="first",inplace=False)
data=data.reset_index()
del data["index"]


#keeping only those words in concatenated comments with length greater than 2 
for i in range(len(data)):
    data.loc[i,"Comments_concat"] = ' '.join([w for w in data.loc[i,"Comments_concat"].split() if len(w)>2])

#stemming and lemmatization on concatenated comments
port = PorterStemmer()
wnl = WordNetLemmatizer()

for i in range(len(data)):    
    data.loc[i,"Comments_concat"]=" ".join([wnl.lemmatize(j) for j in data.loc[i,"Comments_concat"].split()])
    data.loc[i,'Comments_concat']=" ".join([port.stem(j) for j in data.loc[i,'Comments_concat'].split()])

#creating separate dataframes 
data_c=data[data["Validation"]==0].reset_index()
data_nc=data[data["Validation"]==1].reset_index()
del data_c["index"]
del data_nc["index"]

#IDF on comments_concat
#creating corpus of sentences 
corpus_c=[]
for i in range(len(data_c)):
    corpus_c.append(data_c.loc[i,"Comments_concat"])

vectorizer = TfidfVectorizer(
                        use_idf=True, # utiliza o idf como peso, fazendo tf*idf
                        norm=None, # normaliza os vetores
                        smooth_idf=False, #soma 1 ao N e ao ni => idf = ln(N+1 / ni+1)
                        sublinear_tf=False, #tf = 1+ln(tf)
                        binary=False,
                        min_df=1, max_df=1.0, max_features=None,
                        strip_accents='unicode', # retira os acentos
                        ngram_range=(1,1), preprocessor=None,stop_words=None, tokenizer=None, vocabulary=None
             )
X = vectorizer.fit_transform(corpus_c)
idf_c= vectorizer.idf_
Idf_c=list(zip(vectorizer.get_feature_names(), idf_c))
Idf_c=pd.DataFrame(Idf_c,columns=['word','idf'])
Idf_c["idf"]=Idf_c["idf"]-1 #This is done beacuse in the code for TfIdf vectorizer the value of Idf is calculated as +1 to the normal IDF value 
Idf_c=Idf_c.sort_values("idf",ascending=False)

#creating corpus of sentences for non-compliant merchants
corpus_nc=[]
for i in range(len(data_nc)):
    corpus_nc.append(data_nc.loc[i,"Comments_concat"])
X = vectorizer.fit_transform(corpus_nc)
idf_nc= vectorizer.idf_
Idf_nc=list(zip(vectorizer.get_feature_names(), idf_nc))
Idf_nc=pd.DataFrame(Idf_nc,columns=['word','idf'])
Idf_nc["idf"]=Idf_nc["idf"]-1
Idf_nc=Idf_nc.sort_values("idf",ascending=False)

#making dataframes for words present only in one df, only in the other sd and in both with their IDF values
df=pd.merge(Idf_nc,Idf_c,on="word",how="outer",indicator=True)
df["_merge"].value_counts()
df_nc=df[df["_merge"]=="left_only"].reset_index()
df_c=df[df["_merge"]=="right_only"].reset_index()
df_both=df[df["_merge"]=="both"].reset_index()
del df_c["index"]
del df_nc["index"]
del df_both["index"]

#creating corpus of words for making TFIDF vectorizer/ creating features 
corpus=[]
for i in range(len(df_both)):
    if(df_both.loc[i,"idf_x"]<=6.010040930  and df_both.loc[i,"idf_y"]>=7.22):
        corpus.append(df_both.loc[i,"word"])
for i in range(len(df_nc)):
    corpus.append(df_nc.loc[i,"word"])
for i in range(len(df_both)):
    if(df_both.loc[i,"idf_x"]<=4.9102008967744855):
        corpus.append(df_bo32th.loc[i,"word"])
corpus_unique = []
for x in corpus:
    if x not in corpus_unique:
        corpus_unique.append(x)


##TFIDF on comments_concat
vectorizer = TfidfVectorizer(vocabulary=corpus_unique)
X = vectorizer.fit_transform(data[data.columns[-5]])
vect_df_reviews = pd.DataFrame(X.todense(),columns = vectorizer.get_feature_names())   


##count vectorizer on bag_words_tags
data=data.fillna("")
bag_tags=[pd.Series(data["concat_tags"]).str.cat(sep=' ')][0].split()             
bag_tags = [x for x in list(set([y.lower() for y in bag_tags])) if x!= 'nan'] 
vectorizer_tags = CountVectorizer(vocabulary=bag_tags)
Z = vectorizer_tags.fit_transform(data[data.columns[-4]])
vect_df_tags = pd.DataFrame(Z.todense(),columns = vectorizer_tags.get_feature_names())

#dummy variables on country 
vect_df_country=pd.get_dummies(data['Country'])
#dummy variable for category 
vect_df_category=pd.get_dummies(data['Category'])
#dummy variable for Sub_Category
vect_df_sub_category=pd.get_dummies(data["Sub_Category"])

##creating one dataframe for data_after_enchant,vect_df_reviews,vect_df_tags
tempo = pd.concat([data,vect_df_country,vect_df_category,vect_df_sub_category,vect_df,vect_df_reviews,vect_df_tags],axis =1)
output = open(r"",'wb')
pickle.dump(tempo,output)
output.close()
tempo.columns
data.columns
#
x=tempo.copy()
x= tempo.iloc[:,11:]
x.fillna(0,inplace=True)
x = x.rename(columns = {'fit': 'fit_feature'})
y = tempo[['Validation']]

#test train 0.3 to 0.7 
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.3,stratify=y)
#exporting x_test,y_test and x_train,y_train
output = open(r"",'wb')
pickle.dump(x_train,output)
output = open(r"",'wb')
pickle.dump(x_test,output)
output = open(r"",'wb')
pickle.dump(y_train,output)
output = open(r"",'wb')
pickle.dump(y_test,output)
output.close()

y_train_array=y_train.values.ravel()
y_test_array=y_test.values.ravel()

#model 
model= ske.RandomForestClassifier(bootstrap=True, criterion='entropy', max_depth=None,
                max_features=50, n_estimators=400,random_state=1)
model.fit(x_train,y_train_array)                                        

prob_train=pd.DataFrame(model.predict_proba(x_train))
prob_test=pd.DataFrame(model.predict_proba(x_test))

prob_test=prob_test.reset_index()
del prob_test["index"]
y_test=y_test.reset_index()
del y_test["index"]
y_train=y_train.reset_index()
del y_train["index"]


prob1=pd.concat([prob_train,y_train],axis=1)
prob2=pd.concat([prob_test,y_test],axis=1)
prob1.to_csv(".csv",index=False)
prob2.to_csv(".csv",index=False)

#feature importance
model_feat_imp = pd.DataFrame(model.feature_importances_)
model_feat_imp['words'] = x_train.columns
model_feat_imp=model_feat_imp.sort_values(0,ascending=False)

#saving the model 
output = open(r"",'wb')
pickle.dump(model,output)
output.close()

#reading saved model 
input1 = open(r"",'rb')
model=pickle.load(input1)







