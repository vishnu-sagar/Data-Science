import pandas as pd
import os

dir=r"C:\Users\vichu\Documents\assign\Inputfiles"
text=pd.read_csv(os.path.join(dir,"NLPdataEx5data_senti_analyze.txt"),sep='\t',names=['Comment'])
print(text.info())

senti_dict = {}
for each_line in open(r"C:\Users\vichu\Downloads\senti_dict (1).txt"):
    #print(each_line)
    word,score = each_line.split('\t')
    senti_dict[word] = int(score)
for word in text['Comment']:
    words=word.lower().split()
    print(' '.join(words))
    senti=sum( senti_dict.get(word, 0) for word in words )
    if senti > 0:
        print("This is a POSITIVE SENTIMENT\n")
    elif senti < 0:
        print("This is a NEGATIVE SENTIMENT\n")
    else:
        print("This is a NEUTRAL SENTIMENT\n")
        