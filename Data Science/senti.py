import pandas as pd
import os
import io



dir=r"C:\\Users\\vichu\\Documents\ML"
census=pd.read_csv(os.path.join(dir,"train_senti.csv"))
print(census.info())


census.isna()

census1=census.dropna()

census1.concat()
print(census.columns)
print(census1['unique_hash'])

census1['new'] = census1.values.sum(axis=1)

census1['new'] = census1.sum(axis=1).astype(int).astype(str)

name = list(census1.columns)
name.remove('unique_hash')


print(name)

for x in name :
    census1['unique_hash']= census1['unique_hash'].map(str)+census1[x].map(str)
    
df= pd.DataFrame(census1,columns=name)

census['unique_hash'].apply(len)

df = census[census['unique_hash'].apply(lambda x: len(x) <= 40)]

census['unique_hash'].apply(lambda x: x.str.len().gt(40))

census.select_dtypes(['object']).apply(lambda x: x.str.len().gt(40)).axis=1


bad_data=census[~(census.unique_hash.str.len() == 40)]
temp=list(bad_data.index)

for x in temp:
    print(x)
    temp1=x-1
    
    
import pandas as pd
import os
import spacy
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.linear_model import LogisticRegression
from sklearn import svm,naive_bayes
from sklearn import metrics
from sklearn.model_selection import train_test_split

punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

dir=r"D:\ML"
sent=pd.read_csv(os.path.join(dir,"train_senti.csv"))
print(sent.info())

sent1=sent['text'].map(str)+sent['drug'].map(str)
sent1=pd.DataFrame(sent1,columns=['text'])
features=['text']
X_train=sent['text']
X_train1=sent.loc[:1,'text']
y_train=pd.DataFrame(sent['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)

bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
X_train1=tfidf_vector.fit_transform(X_train1)
print(X_train1.toarray())
df1=spacy_tokenizer("Autoimmune diseases tend to come in clusters. As for Gilenya – if you feel good, don’t think about it, it won’t change anything but waste your time and energy. I’m taking Tysabri and feel amazing, no symptoms (other than dodgy color vision, but I’ve had it since always, so, don’t know) and I don’t know if it will last a month, a year, a decade, ive just decided to enjoy the ride, no point in worrying.gilenya")
classifier = LogisticRegression(multi_class='ovr')

classifier1=svm.SVC(decision_function_shape='ovr')
classifier2=naive_bayes.BernoulliNB()
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', tfidf_vector),
                 ('classifier', classifier1)])
    


# model generation
pipe.fit(X_train,y_train)



'''
sent_test=pd.read_csv(os.path.join(dir,"test_senti.csv"))
print(sent_test.info())

X_test=sent_test['text']

y_test=pipe.predict(X_test)
'''

predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted,average='macro'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted,average='macro'))
print("Logistic Regression Recall:",metrics.f1_score(y_test, predicted,average='macro'))
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test, predicted))

import os
import pandas as pd
import numpy as np
import re
import spacy
import nltk
from bs4 import BeautifulSoup
from nltk import word_tokenize          
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from sklearn.linear_model import LogisticRegression
from sklearn import svm,naive_bayes,model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, Bidirectional
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

data_dir = r"C:\\Users\\vichu\\Documents\ML"
glove_file = r"C:\\Users\\vichu\\Documents\ML\glove.6B.50d.txt"
word_embed_size = 50
batch_size = 64
epochs = 1
seq_maxlen = 80
nltk.download('punkt')
nltk.download('stopwords')
'''
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()
'''
def cleanReview(review):        
        #1.Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #2.Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #3.Convert words to lower case
        review_text = review_text.lower()
        #4.remove stop words
        review_words = word_tokenize(review)
        words = [w for w in review_words if not w in stopwords.words("english")]
        return ' '.join(words)
    
def buildVocabulary(reviews):
    tokenizer = Tokenizer(lower=False, split=' ')
    tokenizer.fit_on_texts(reviews)
    return tokenizer

def getSequences(reviews, tokenizer, seq_maxlen):
    reviews_seq = tokenizer.texts_to_sequences(reviews)
    return np.array(pad_sequences(reviews_seq, maxlen=seq_maxlen))

def loadGloveWordEmbeddings(glove_file):
    embedding_vectors = {}
    f = open(glove_file,encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        value = np.asarray(values[1:], dtype='float32')
        embedding_vectors[word] = value
    f.close()
    return embedding_vectors

def getEmbeddingWeightMatrix(embedding_vectors, word2idx):    
    embedding_matrix = np.zeros((len(word2idx)+1, word_embed_size))
    for word, i in word2idx.items():
        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

'''
imdb_train = pd.read_csv(os.path.join(data_dir,"labeledTrainData.tsv"), header=0, 
                    delimiter="\t", quoting=3)
imdb_train.shape
imdb_train.info()
imdb_train.loc[0:10,'review']
'''
dir=r"C:\\Users\\vichu\\Documents\ML"
sent=pd.read_csv(os.path.join(dir,"train_senti.csv"))
print(sent.info())

sent1=sent['text'].map(str)+sent['drug'].map(str)
sent1=pd.DataFrame(sent1,columns=['text'])

X_train=sent1['text']
y_train=pd.DataFrame(sent['sentiment'])
sent['text'][0:4]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)
y_test1=y_test[0:50]
#preprocess text
train_clean = X_train[0:100].map(cleanReview)
test_clean=X_test[0:50].map(cleanReview)
print(len(train_clean))

#build vocabulary over all reviews
tokenizer = buildVocabulary(train_clean)
vocab_size = len(tokenizer.word_index) + 1
print(tokenizer.word_index)
print(vocab_size)

X_train = getSequences(train_clean, tokenizer, seq_maxlen)
y_train = np_utils.to_categorical(sent['sentiment'][0:100])
y_test1=np_utils.to_categorical(y_test1)


#load pre-trained word embeddings
embedding_vectors = loadGloveWordEmbeddings(glove_file)
print(len(embedding_vectors))
#get embedding layer weight matrix
embedding_weight_matrix = getEmbeddingWeightMatrix(embedding_vectors, tokenizer.word_index)
print(embedding_weight_matrix.shape)

'''
dir=r"C:\\Users\\vichu\\Documents\ML"
sent=pd.read_csv(os.path.join(dir,"train_senti.csv"))
print(sent.info())

sent.loc[0:4,'text']

sent1=sent['text'].map(str)+sent['drug'].map(str)
sent1=pd.DataFrame(sent1,columns=['text'])
features=['text']
X_train=sent1['text']
X_train1=sent.loc[:1,'text']
y_train=pd.DataFrame(sent['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25)


bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)
#X_train1=tfidf_vector.fit_transform(X_train1)
#print(X_train1.toarray())
#df1=spacy_tokenizer("Autoimmune diseases tend to come in clusters. As for Gilenya – if you feel good, don’t think about it, it won’t change anything but waste your time and energy. I’m taking Tysabri and feel amazing, no symptoms (other than dodgy color vision, but I’ve had it since always, so, don’t know) and I don’t know if it will last a month, a year, a decade, ive just decided to enjoy the ride, no point in worrying.gilenya")
classifier = LogisticRegression(class_weight='balanced')

classifier1=svm.SVC(decision_function_shape='ovr')
classifier2=naive_bayes.BernoulliNB()
# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', tfidf_vector)])
    
X_train=pipe.fit_transform(X_train)
'''
#build model        
input = Input(shape=(X_train.shape[1],))

inner = Embedding(input_dim=vocab_size, output_dim=word_embed_size, 
                   input_length=seq_maxlen, weights=[embedding_weight_matrix], 
                   trainable = False) (input)
inner = Conv1D(64, 5, padding='valid', activation='relu', strides=1)(inner)
inner = MaxPooling1D(pool_size=4)(inner)
inner = LSTM(100, return_sequences=False)(inner)
inner = Dropout(0.3)(inner)
inner = Dense(50, activation='relu')(inner)
output = Dense(3, activation='softmax')(inner)

model = Model(inputs = input, outputs = output)
model.compile(Adam(lr=0.01), 'categorical_crossentropy', metrics=['accuracy'])

save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train, verbose=1, epochs=epochs, batch_size=batch_size, 
                    callbacks=[save_weights], validation_split=0.1)

X_test = getSequences(test_clean, tokenizer, seq_maxlen)
y_pred=model.predict(X_test).argmax(axis=-1)



print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test1, y_pred))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted,average='macro'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted,average='macro'))
print("Logistic Regression Recall:",metrics.f1_score(y_test1,y_pred,average='macro'))
print(metrics.classification_report(y_test, predicted))
print(metrics.confusion_matrix(y_test1,y_pred))

imdb_test = pd.read_csv(os.path.join(data_dir,"testData.tsv"), header=0, 
                    delimiter="\t", quoting=3)
imdb_test.shape
imdb_test.info()
imdb_test.loc[0:4,'review']

#preprocess text
review_test_clean = imdb_test['review'][0:4].map(cleanReview)
print(len(review_test_clean))

X_test = getSequences(review_test_clean, tokenizer, seq_maxlen)
imdb_test['sentiment'] = model.predict(X_test).argmax(axis=-1)
imdb_test.to_csv(os.path.join(data_dir,'submission.csv'), columns=['id','sentiment'], index=False)
