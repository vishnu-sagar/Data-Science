import pandas as pd
import os
from nltk import sent_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

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
    
sent=[]
for n in text['Comment']:
    sent.append(sent_tokenize(n))
    
data_out=pd.DataFrame((list(traverse(sent))),columns=['token_Comment']) 

data_out.to_csv(os.path.join(dir,'data_out.csv'), index=False)

chunk_sentence = ne_chunk(pos_tag(data_out['token_Comment']))

chunk_sentence.draw()



#word token




