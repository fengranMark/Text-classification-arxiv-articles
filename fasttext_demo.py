#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm, naive_bayes


# In[2]:


train = pd.read_csv('train.csv', sep=',', usecols=[0,1,2])
headers = list(train.columns.values)


# In[3]:


# category2idx
category_list = []
category2idx = {}
idx = 1
for key in train[headers[-1]]:
    if key not in category_list:
        category_list.append(key)
        category2idx[key] = idx
        idx += 1
print(category2idx)


# In[4]:


# load corpus and convert label
Id = list(train[headers[0]])
raw_corpus = list(train[headers[1]])
category = list(train[headers[2]])
raw_corpus = [line.lower().strip().replace('\n', ' ') for line in raw_corpus]
labels = [category2idx[key] for key in category]


# In[5]:


# preprocessing
def Stem(corpus):
    stemmer = PorterStemmer()
    new_corpus = []
    for sent in corpus:
        sent_words = sent.split(' ')
        words_stem = [stemmer.stem(word) for word in sent_words]
        new_sent = ' '.join(words_stem)
        new_corpus.append(new_sent)
    print("stem finished")
    return new_corpus

def Lemma(corpus):
    lemmatizer = WordNetLemmatizer()
    new_corpus = []
    for sent in corpus:
        sent_words = sent.split(' ')
        words_lemma = [lemmatizer.lemmatize(word) for word in sent_words]
        new_sent = ' '.join(words_lemma)
        new_corpus.append(new_sent)
    print("lemma finished")
    return new_corpus


# In[6]:


def remove_special(corpus):
    keylist = [chr(i) for i in range(97, 123)]
    keylist.append(' ')
    special = []
    new_corpus = []
    for sent in corpus:
        sent = sent.lower()
        for key in sent:
            if key not in keylist:
                special.append(key)
                sent = sent.replace(key, ' ')
        new_corpus.append(sent)
    
    return new_corpus


# In[7]:


def remove_stopwords(corpus):
    new_corpus = []
    stopwords = open('stopwords.txt', 'r')
    #stopwords = stopwords.readlines()
    stopwords = [line.strip() for line in stopwords]
    for sent in corpus:
        sent_words = sent.split(' ')
        cleaned_words = [word for word in sent_words if word not in stopwords]
        new_sent = ' '.join(cleaned_words)
        new_corpus.append(new_sent)
    print("remove stopwords finished")
    return new_corpus


# In[8]:


x = remove_special(raw_corpus)
#x = Stem(x)
print("data shuffling...")
x = np.array(x) 
# y = np.array(labels)
y = np.array(category)
index = np.arange(len(y))  
np.random.shuffle(index)
x = x[index]
y = y[index]

# split
print("data spliting...")
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
print('x_train.shape : ', x_train.shape)
print('x_valid.shape : ', x_valid.shape)


# In[9]:


def preprocess_fasttext(x_train, x_valid, y_train, y_valid, train_path, valid_path):
    train_list, valid_list = [], []
    for x, y in zip(x_train, y_train):
        line_train = '__label__' + y + ' ' + x + '\n'
        train_list.append(line_train)
    for x, y in zip(x_valid, y_valid):
        line_valid = '__label__' + y + ' ' + x + '\n'
        valid_list.append(line_valid)
    with open(train_path, 'w') as f:
        f.writelines(train_list)
    with open(valid_path, 'w') as f:
        f.writelines(valid_list)
train_path = 'train.txt'
valid_path = 'valid.txt'
preprocess_fasttext(x_train, x_valid, y_train, y_valid, train_path, valid_path)


# In[10]:


import fasttext

def train_fasttext(train_path):
    ws = 5
    minCount = 2
    minn = 2
    maxn = 20
    dim = 256
    lr = 0.5
    neg = 5
    t = 0.0005
    epoch = 50
    wordNgrams = 4
    pretrainedVectors = None
    model = fasttext.train_supervised(input=train_path, dim=dim, ws=ws, minCount=minCount, lr=lr,
                                      minn=minn, maxn=maxn, neg=neg, t=t, epoch=epoch, wordNgrams=wordNgrams)
    return model

model = train_fasttext(train_path)


# In[11]:


def test_fasttext(valid_path, model):
    correct = 0
    with open(valid_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        s = line.split()
        text, label = ' '.join(s[1:]), s[0]
        pred = model.predict(text)[0][0]
        if pred == label:
            correct += 1
    acc = correct / len(lines)
    print(f'total number: {len(lines)}')
    print(f'correct predictions: {correct}')
    print(f'acc: {acc}')
test_fasttext(valid_path, model)

