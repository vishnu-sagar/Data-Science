import pandas as pd
import os
import spacy
import string
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Input, Bidirectional
from keras.models import Model,Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()

def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return ' '.join(mytokens)

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


dir=r"D:\ML"
sent=pd.read_csv(os.path.join(dir,"train_senti.csv"))
print(sent.info())

glove_file = r"D:\ML\glove.6B.50d.txt"
word_embed_size = 50
batch_size = 64
epochs =20
seq_maxlen = 100

sent_join=sent['text'].map(str)+sent['drug'].map(str)
sent_join=pd.DataFrame(sent_join,columns=['text'])

X=sent_join['text']
y=sent['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)


X=X_train.map(clean_text)
X=X.map(spacy_tokenizer)


tokenizer = buildVocabulary(X)
vocab_size = len(tokenizer.word_index) + 1
print(tokenizer.word_index)
print(vocab_size)


X_train = getSequences(X, tokenizer, seq_maxlen)
y_train = np_utils.to_categorical(y_train)


embedding_vectors = loadGloveWordEmbeddings(glove_file)
print(len(embedding_vectors))
#get embedding layer weight matrix
embedding_weight_matrix = getEmbeddingWeightMatrix(embedding_vectors, tokenizer.word_index)
print(embedding_weight_matrix.shape)

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
model.compile(Adam(lr=0.0000001), 'categorical_crossentropy', metrics=['accuracy'])

save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(X_train, y_train, verbose=1, epochs=epochs, batch_size=batch_size, 
                    callbacks=[save_weights], validation_split=0.1)



X_test=X_test.map(clean_text)
X_test=X_test.map(spacy_tokenizer)

X_test = getSequences(X_test, tokenizer, seq_maxlen)

y_pred=model.predict(X_test).argmax(axis=-1)

print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Logistic Regression Precision:",metrics.precision_score(y_test, y_pred,average='macro'))
print("Logistic Regression Recall:",metrics.recall_score(y_test, y_pred,average='macro'))
print("f1_score:",metrics.f1_score(y_test, y_pred,average='macro'))
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))



