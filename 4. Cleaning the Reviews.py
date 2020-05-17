# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
from fuzzywuzzy import fuzz
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
#import enchant
#from enchant.checker import SpellChecker

df = pd.read_csv(r".csv", low_memory=False,encoding="latin-1")  ###Extracted output path
df=df.drop_duplicates(subset=['ID'], keep='first', inplace=False) ##Duplicates removal
temp=df.copy()
temp=temp.fillna("")

bag_words=pd.read_csv(r"")  ##bag of words path
lst = list(bag_words)


#stopwords removal from concat_reviews
m = []
for i in range(len(temp)):
    x = " ".join(re.split("[^\w\s]",temp.loc[i,"concat_reviews"])) ## to be changed
    tokens = x.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    c = ' '.join(words)
    m.append(c)
temp['new_concat'] = pd.DataFrame(m)

###Spelling correction 
d = enchant.DictWithPWL("en", "") ###reading bag_of_words.txt
chkr = SpellChecker(d)

for i in range(len(temp)):
    print(i)
    chkr.set_text(temp["new_concat"][i])
    for err in chkr:
        for j in range(len(d.suggest(err.word))):
            for k in range(len(lst)):
                if(fuzz.ratio(err.word,lst[k])>85):
                    temp['Comments'][i] = temp["new_concat"][i].replace(err.word,lst[k])
                    print("found: ", err.word, "\t",lst[k], "\t",fuzz.ratio(err.word,lst[k]))

