# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from googletrans import Translator
trans = Translator()

df_main_input = pd.read_csv(".csv",encoding='latin-1') #reading extracted output of Non-English geography

df_main_input = df_main_input.fillna("") ##na replaced with blanks

#reviews concatenation
df_main_input['concat_reviews'] = df_main_input[df_main_input.columns[16:21]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
df2=df_main_input.copy()

##creating a column dummy with values as dummy 
df2['dummy'] = 'dummy'
df2.reset_index(inplace=True)
del df2['index']

###Splitting reviews in 1300 characters length
df2['concat_reviews1']=df2.concat_reviews.str[0:1299]
df2['concat_reviews2']=df2.concat_reviews.str[1300:2599]
df2['concat_reviews3']=df2.concat_reviews.str[2600:3899]
df2['concat_reviews4']=df2.concat_reviews.str[3900:5199]
df2['concat_reviews5']=df2.concat_reviews.str[5200:6399]
df2['concat_reviews6']=df2.concat_reviews.str[6400:]

##translation
error_list_big=[]
for i in range(len(df2)):
    a = ''
    for j in ['concat_reviews1','concat_reviews2', 'concat_reviews3', 'concat_reviews4', 'concat_reviews5', 'concat_reviews6']:
        try:
            a = a+(trans.translate(df2[j][i], dest = 'en').text)
            df2['dummy'][i] = a
            print(i)
        except Exception as error:
            error_list_big.append(error)
        continue

##checking number of non-translated reviews
df2[df2["dummy"]=="dummy"]

##drop columns
df2.drop(['concat_reviews1','concat_reviews2', 'concat_reviews3', 'concat_reviews4', 'concat_reviews5', 'concat_reviews6'],axis=1,inplace=True)

df2=df2.reset_index()
del df2["index"]

##Replacing non-translated reviews with concat_reviews
for i in range(len(df2)):
    if df2["dummy"][i]=="dummy":
        print(i)
        df2["dummy"][i]=df2["concat_reviews"][i]

###Substituting translated reviews in concat_reviews column		
df2["concat_reviews"]=df2["dummy"]        
##deleting dummy column
del df2["dummy"]         
df2.to_csv("translated.csv",index=None) ##writing translated file

