import pandas as pd
import os
from nltk import word_tokenize
import string
import re

punct=string.punctuation
dir=r"C:\\Users\\vichu\\Documents\assign\Inputfiles"
text=pd.read_csv(os.path.join(dir,"data_in.csv"),error_bad_lines=False)
print(text.info())

def traverse(o, tree_types=(list, tuple)):
    if isinstance(o, tree_types):
        for value in o:
            for subvalue in traverse(value, tree_types):
                yield subvalue
    else:
        yield o

word_token=[]
for n in text['Comment']:
    word_token.append(word_tokenize(n))
    
my_words=list(traverse(word_token))

words = [ re.sub("[^a-zA-Z]","", word) for word in my_words if word not in punct]
 
def clean_words(x): 
    words = re.sub("[^a-zA-Z]"," ", x)
    return words

words=clean_words(words)

    

    
data_out_words=pd.DataFrame(words,columns=['words_Comment']) 

data_out_words=data_out_words[data_out_words.words_Comment !=""] 

data_out_words.to_csv(os.path.join(dir,'data_out.csv'), index=False)
    





