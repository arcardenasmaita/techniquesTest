
# https://www.kaggle.com/teesoong/explainable-ai-on-a-nlp-lstm-model-with-lime
# Explainable AI on a NLP LSTM model with LIME


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Load Data
df = pd.read_csv('data/bitcointweets.csv', header=None)
df = df[[1, 7]]
df.columns = ['tweet', 'label']
df.head()

# inspect sentiment
sns.countplot(df['label'])
# Show the plot
plt.show()

##################################
##################################

# text length
df['text_length'] = df['tweet'].apply(len)
print(df[['label', 'text_length', 'tweet']].head())

g = sns.FacetGrid(df,col='label')
g.map(plt.hist, 'text_length')
plt.show()

##################################
##################################

# We are going to clean up the tweets, remove special chars, stop words, URL links, etc..
from nltk.corpus import stopwords # OJO en consola: nltk.download('stopwords')
from wordcloud import WordCloud
import re

def clean_text(s):
    s = re.sub(r'http\S+', '', s)
    s = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', s)
    s = re.sub(r'@\S+', '', s)
    s = re.sub('&amp', ' ', s)
    return s
df['clean_tweet'] = df['tweet'].apply(clean_text)

text = df['clean_tweet'].to_string().lower()
wordcloud = WordCloud(
    collocations=False,
    relative_scaling=0.5,
    stopwords=set(stopwords.words('english'))).generate(text)

plt.figure(figsize=(12,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Encode Categorical Variable
X = df['clean_tweet']
# y = pd.get_dummies(df['label']).values
encode_cat = {"label":     {"['neutral']": 0, "['positive']": 1, "['negative']": 2}}
y_df = df.replace(encode_cat)
y = y_df['label']
print(y.value_counts())

seed = 101 # fix random seed for reproducibility
np.random.seed(seed)

# Split Train Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=seed)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1) # new

vocab_size = 20000  # Max number of different word, i.e. model input dimension
maxlen = 80  # Max number of words kept at the end of each text

##################################
##################################
# we need to create our LSTM model and use the KerasClassifier in keras.wrappers.scikit_learn.
# Reason is that to use LIME text explainer, we need to use a sklearn pipeline

import keras.preprocessing.text
from keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import TransformerMixin
from sklearn.base import BaseEstimator

class TextsToSequences(keras.preprocessing.text.Tokenizer, BaseEstimator, TransformerMixin):
    """ Sklearn transformer to convert texts to indices list
    (e.g. [["the cute cat"], ["the dog"]] -> [[1, 2, 3], [1, 4]])"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, texts, y=None):
        self.fit_on_texts(texts)
        return self

    def transform(self, texts, y=None):
        return np.array(self.texts_to_sequences(texts))

sequencer = TextsToSequences(num_words=vocab_size)

##################################
##################################

class Padder(BaseEstimator, TransformerMixin):
    """ Pad and crop uneven lists to the same length.
    Only the end of lists longernthan the maxlen attribute are
    kept, and lists shorter than maxlen are left-padded with zeros

    Attributes
    ----------
    maxlen: int
        sizes of sequences after padding
    max_index: int
        maximum index known by the Padder, if a higher index is met during
        transform it is transformed to a 0
    """

    def __init__(self, maxlen=500):
        self.maxlen = maxlen
        self.max_index = None

    def fit(self, X, y=None):
        self.max_index = pad_sequences(X, maxlen=self.maxlen).max()
        return self

    def transform(self, X, y=None):
        X = pad_sequences(X, maxlen=self.maxlen)
        X[X > self.max_index] = 0
        return X


padder = Padder(maxlen)

##################################
##################################

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline

batch_size = 128
max_features = vocab_size + 1

import tensorflow as tf
tf.random.set_seed(seed)

def create_model(max_features):
    """ Model creation function: returns a compiled LSTM"""
    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

##################################
##################################

# Use Keras Scikit-learn wrapper to instantiate a LSTM with all methods
# required by Scikit-learn for the last step of a Pipeline
sklearn_lstm = KerasClassifier(build_fn=create_model, epochs=2, batch_size=batch_size,
                               max_features=max_features, verbose=1)

# Build the Scikit-learn pipeline
pipeline = make_pipeline(sequencer, padder, sklearn_lstm)
pipeline.fit(X_train, y_train);

##################################
##################################
print('Computing predictions on test set...')
y_preds = pipeline.predict(X_test)

##################################
##################################

##################################
##################################

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def model_evaluate():
    print('Test Accuracy:\t{:0.1f}%'.format(accuracy_score(y_test, y_preds) * 100))

    # classification report
    print('\n')
    print(classification_report(y_test, y_preds))

    # confusion matrix
    confmat = confusion_matrix(y_test, y_preds)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

##################################
##################################
print(model_evaluate())

##################################
##################################
# our LSTM model seems very accurate on the test set. Now for the interesting part in using LIME.
# We choose a sample from test set
idx = 15
test_text = np.array(X_test)
test_class = np.array(y_test)
text_sample = test_text[idx]
class_names = ['neutral', 'positive', 'negative']
print(text_sample)
print('Probability =', pipeline.predict_proba([text_sample]).round(3))
print('True class: %s' % class_names[test_class[idx]])

##################################
##################################
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(text_sample, pipeline.predict_proba, num_features=6, top_labels=2)
exp.show_in_notebook(text=text_sample)

##################################
##################################
# The word 'successful' is a major reason for the positive classification, let's see what happens when we remove it!
text_sample2 = re.sub('successful', ' ', text_sample)
print(text_sample2)

##################################
##################################
# Let's rerun the prediction:
print('Probability =', pipeline.predict_proba([text_sample2]).round(3))

# Wow, as you can see, when you remove 'successful', the LSTM model class probabilities have indeed changed!