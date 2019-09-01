import pandas as pd
import os
from nltk.tag import pos_tag
from nltk import word_tokenize,ne_chunk
import string

punct=string.punctuation
dir=r"C:\\Users\\vichu\\Documents\assign\Inputfiles"
text=pd.read_csv(os.path.join(dir,"NLPdataEx3&4-data_in.txt"),sep='\t',names=['Comment'])
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

words = [ word for word in my_words if word not in punct]

tree=ne_chunk(pos_tag(words))
tree.draw()
    
