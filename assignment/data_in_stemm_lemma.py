import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd
import os
from nltk.stem import PorterStemmer,WordNetLemmatizer


def traverse(o, tree_types=(list, tuple)):
    if isinstance(o, tree_types):
        for value in o:
            for subvalue in traverse(value, tree_types):
                yield subvalue
    else:
        yield o

import string

punct=string.punctuation

dir=r"C:\\Users\\vichu\\Documents\assign\Inputfiles"
text=pd.read_csv(os.path.join(dir,"NLPdataEx3&4-data_in.txt"),sep='\t',names=['Comment'])
print(text.info())


word_token=[]
for n in text['Comment']:
    word_token.append(word_tokenize(n))
    
my_words=list(traverse(word_token))
    
words = [ word for word in my_words if word not in punct]

words_clean = [w for w in words if not w in stopwords.words("english")]
    

word_stem=[]
ps=PorterStemmer()
for w in words_clean:
    word_stem.append(ps.stem(w))
    
    
word_lemma=[]
lm=WordNetLemmatizer()
for x in words_clean:
    word_lemma.append(lm.lemmatize(x))
    
    
    
fmt = '{:<8}{:<20}{}'

print("COMPARING STEMMING AND LEMMATIZATION")
print(fmt.format('', 'STEMMING', 'LEMMATIZATION'))
for i, (x, n) in enumerate(zip(word_stem, word_lemma)):
    print(fmt.format(i, x, n))
    


