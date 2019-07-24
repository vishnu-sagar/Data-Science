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

