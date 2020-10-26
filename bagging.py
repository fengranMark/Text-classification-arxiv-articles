#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import codecs
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn import svm, naive_bayes
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier
from random import choice
import random


# In[2]:


train = pd.read_csv('train.csv', sep=',', usecols=[0,1,2])
headers = list(train.columns.values)
# category2idx 
category_list = []
category2idx = {}
idx = 0
for key in train[headers[-1]]:
    if key not in category_list:
        category_list.append(key)
        category2idx[key] = idx
        idx += 1
#print(category2idx)


# In[3]:


# load corpus and convert label
train_Id = list(train[headers[0]])
train_corpus = list(train[headers[1]])
category = list(train[headers[2]])
train_corpus = [line.lower().strip().replace('\n', ' ') for line in train_corpus]
labels = [category2idx[key] for key in category]


# In[4]:


test = pd.read_csv('test.csv', sep=',', usecols=[0,1])
headers_ = list(test.columns.values)
test_Id = list(test[headers_[0]])
test_corpus = list(test[headers_[1]])
test_corpus = [line.lower().strip().replace('\n', ' ') for line in test_corpus]


# In[5]:


stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
special = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*',
          '+', '\\', '•', '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹',
          '≤', '‡', '√', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def tokenize(corpus):
    new_corpus = []
    for line in corpus:
        post = str(line)
        #clean special
        for punct in special:
            post = post.replace(punct, ' ')
        word_list = post.split()
        # use stem
        # to lower case
        #filtered_words = [stemmer.stem(word.lower()) for word in word_list]
        #filtered_words = [stemmer.stem(word.lower()) for word in word_list if word not in stop_words]
        filtered_words = [word for word in word_list if word not in stop_words]
        #filtered_words = [lemmatizer.lemmatize(word) for word in word_list if word not in stop_words]

        #remmove the words shorter than 3 chars
        #filtered_words = [word for word in filtered_words if len(word) >= 3]
        new_sent = ' '.join(filtered_words)
        new_corpus.append(new_sent)
    return new_corpus


# In[6]:


print("data shuffling...")
x = np.array(train_corpus) 
y = np.array(labels)
index = np.arange(len(y))  
np.random.shuffle(index)
x = x[index]
y = y[index]

# split
print("data spliting...")
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
print('x_train.shape : ', x_train.shape)
print('x_valid.shape : ', x_valid.shape)

# preprocessing
vocab_corpus = train_corpus + test_corpus
vocab_corpus = tokenize(vocab_corpus)
x_train = tokenize(x_train)
x_valid = tokenize(x_valid)


# In[7]:

tfidf_model1 = TfidfVectorizer(stop_words = 'english', min_df = 2, max_df = 0.75, ngram_range=(1,2))
#tfidf_model2 = TfidfVectorizer(stop_words = 'english', min_df = 2, max_df = 0.5, ngram_range=(1,2))
x_all1 = tfidf_model1.fit_transform(vocab_corpus)
x_train1 = tfidf_model1.transform(x_train)
x_valid1 = tfidf_model1.transform(x_valid)
    
#x_all2 = tfidf_model2.fit_transform(vocab_corpus)
#x_train2 = tfidf_model2.transform(x_train)
#x_valid2 = tfidf_model2.transform(x_valid)
#print('x_train.shape : ', x_train.shape)
#print('x_valid.shape : ', x_valid.shape)





# In[15]:

classifiers = {
    
    #'Logistic (Multinomial)': LogisticRegression(C = 1, penalty='l2', solver='saga', multi_class='multinomial'),
    #'Multinomial NB': naive_bayes.MultinomialNB(alpha = 0.11),
    #'Bernoulli NB':naive_bayes.BernoulliNB(alpha = 0.05),
    #'Linear SVC': svm.LinearSVC(C = 0.81),
    #'Boosting NB': AdaBoostClassifier(base_estimator=naive_bayes.MultinomialNB(alpha = 0.11), n_estimators=100),
    #'Boosting SVC': AdaBoostClassifier(base_estimator=CalibratedClassifierCV(svm.LinearSVC(C = 0.81)), n_estimators=20),
    #'Boosting-Bagging NB': BaggingClassifier(base_estimator=AdaBoostClassifier(base_estimator=naive_bayes.MultinomialNB(alpha = 0.11), 
    #                                                                           n_estimators=10), n_estimators=10),
    #'Boosting-Bagging SVC': BaggingClassifier(base_estimator=AdaBoostClassifier(base_estimator=CalibratedClassifierCV(svm.LinearSVC(C = 0.81)), 
    #                                                                           n_estimators=10), n_estimators=10),                                                                   
    #'Boosting_tree': XGBClassifier(n_estimators=10, objective='multi:softmax', num_class=15, max_depth=10),
    #'Boosting_linear': XGBClassifier(booster='gblinear', n_estimators=10),
    'Bagging NB2': BaggingClassifier(naive_bayes.BernoulliNB(alpha = 0.001), n_estimators=10), # 0.06
    'Bagging NB': BaggingClassifier(naive_bayes.MultinomialNB(alpha = 0.05), n_estimators=10), # 0.1
    'Bagging SVC': BaggingClassifier(base_estimator=CalibratedClassifierCV(svm.LinearSVC(C = 1)), n_estimators=10), # 1.04
    'Bagging LR': BaggingClassifier(base_estimator=LogisticRegression(C = 100, max_iter=100, solver='saga', multi_class='multinomial'), n_estimators=10), # 1.08
}
# 0.11 0.81 1.09


# In[ ]:


clf_probas = []
for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(x_train1, y_train)
    y_pred = classifier.predict(x_valid1)
    accuracy = metrics.accuracy_score(y_valid, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    probas = classifier.predict_proba(x_valid1)
    
    y_pred = classifier.predict(x_valid)
    accuracy = metrics.accuracy_score(y_valid, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    scores = cross_val_score(classifier, x_train1, y_train, cv=5, scoring='accuracy')
    cv_scores = scores.mean()
    print("Accuracy (cv) for %s: %0.1f%% " % (name, cv_scores * 100))

    # View probabilities:
    probas = classifier.predict_proba(x_valid)
    clf_probas.append(probas)
print(len(clf_probas))

# In[11]:


y_pred_bag = []
for i in range(clf_probas[0].shape[0]): #1500
    # 15个类别中第i个类别的概率和
    y_prob_i = []
    for j in range(clf_probas[0].shape[1]): #15
        prob_sum = 0
        for k in range(len(clf_probas)):
            prob_sum += clf_probas[k][i][j]
        y_prob_i.append(prob_sum)
    max_index = y_prob_i.index(max(y_prob_i, key = abs))
    y_pred_bag.append(max_index)
    
new_accuracy = metrics.accuracy_score(y_valid, y_pred_bag)
print("Bagging Accuracy: %0.1f%% " % (new_accuracy * 100))


# In[14]:



x_train_valid = tokenize(x)
x_train_valid1 = tfidf_model1.transform(x_train_valid)
#x_train_valid2 = tfidf_model2.transform(x_train_valid)
x_test = tokenize(test_corpus)
x_test = np.array(x_test) 
x_test1 = tfidf_model1.transform(x_test)
#x_test2 = tfidf_model2.transform(x_test)

clf_probas_test = []
for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(x_train_valid1, y)
    probas_test = classifier.predict_proba(x_test1)
    clf_probas_test.append(probas_test)
print(len(clf_probas_test))
    #print(probas[:5, :])

y_test = []
for i in range(clf_probas_test[0].shape[0]): #1500
    # 15个类别中第i个类别的概率和
    y_prob_i = []
    for j in range(clf_probas_test[0].shape[1]): #15
        prob_sum = 0
        for k in range(len(clf_probas_test)):
            prob_sum += clf_probas_test[k][i][j]
        y_prob_i.append(prob_sum)
    max_index = y_prob_i.index(max(y_prob_i, key = abs))
    y_test.append(max_index)
    
y_test_cat = []
for label in y_test:
    for value in category2idx:
        if category2idx[value] == label:
            y_test_cat.append(value)
            break
print(len(y_test_cat))
submisstion = pd.DataFrame({'Id':test_Id,'Category':y_test_cat})
submisstion.to_csv("sub_10.19_re4bag10.csv", index = False, sep=',')



