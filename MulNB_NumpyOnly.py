#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import random
random.seed(3395)

# initial resources for preprocessing
# a special charactor list and a stopwords txt file
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

stopwords = open('stopwords.txt', 'r', encoding = 'utf-8')
stopwords = [line.strip() for line in stopwords]

# transform category type from str to int. (e.g astro-ph -> 1)
def cat2idx(classes):
    """
    classes : list (str type of category)
    returns : dictionary (labels mapping)
    """
    category_list = []
    category2idx = {}
    idx = 1
    for key in classes:
        if key not in category_list:
            category_list.append(key)
            category2idx[key] = idx
            idx += 1
    return category2idx

# preprocessing: remove special characters, stopwords
def preprocess(corpus):
    """
    corpus : list (str of each abstract)
    returns : list (each abstract after preprocessing)
    """
    new_corpus = []
    for line in corpus:
        post = str(line)
        #clean special
        for punct in special:
            post = post.replace(punct, ' ')
        word_list = post.split()
        filtered_words = [word for word in word_list if word not in stopwords]

        #remmove the words shorter than 3 chars
        #filtered_words = [word for word in filtered_words if len(word) >= 3]
        new_sent = ' '.join(filtered_words)
        new_corpus.append(new_sent)
    return new_corpus

# dataload function
def load_data(datapath):
    """
    datapath : str (corpus path) 
    returns : 
    train mode : 
        Id : list 
        corpus : numpy array of shape (number of training examples, )  
        labels : numpy array of shape (number of training examples, )
        category2idx : dictinary
    test mode:
        Id : list 
        corpus : numpy array of shape (number of test examples, )  
    """
    data = pd.read_csv(datapath, sep = ',')
    headers = list(data.columns.values)
    if len(headers) > 2:
        mode = 'train'
    else:
        mode = 'test'
    Id = list(data[headers[0]])
    corpus = list(data[headers[1]])
    corpus = [line.lower().strip().replace('\n', ' ') for line in corpus]
    corpus = preprocess(corpus)
    corpus = np.array(corpus)
    if mode == 'train':
        category = list(data[headers[2]])
        category2idx = cat2idx(category)
        labels = [category2idx[key] for key in category]
        labels = np.array(labels)
        return Id, corpus, labels, category2idx
    else:
        return Id, corpus


# Multinational Naive Bayes MaxLikelihood calculation
class MultinationalMaxLikelihood:
    def __init__(self):
        # smoothing value
        self.alpha = 1
        if self.alpha < 0:
            print('alpha should be > 0.')

    def train(self, train_data):
        # calculate N_yi, N_y and theta
        """
        train_data : numpy array of shape (number of train examples, number of features)
        """
        
        self.N_yi = np.sum(train_data, axis = 0) + self.alpha
        self.N_y = np.sum(train_data) + self.alpha * train_data.shape[1] # mul
        self.theta = np.log(self.N_yi) - np.log(self.N_y)

    def loglikelihood(self, test_data):
        # calculate each test sample loglikelihood
        """
        test_data : numpy array of shape (number of test examples, number of features)
        return : numpy array of shape (test_data.shape[0])
        """
        
        log_prob = np.dot(test_data, self.theta)
        return log_prob


# BayesClassifier
class BayesClassifier:
    def __init__(self, maximum_likelihood_models, priors):
        self.maximum_likelihood_models = maximum_likelihood_models
        self.priors = priors
        if len(self.maximum_likelihood_models) != len(self.priors):
            print('The number of ML models must be equal to the number of priors!')
        self.n_classes = len(self.maximum_likelihood_models)

    # Returns a matrix of size number of test ex. times number of classes containing the log
    # probabilities of each test example under each model, trained by ML.
    def loglikelihood(self, test_data):
        """
        test_data : numpy array of shape (number of test examples, number of features)
        return : numpy array of shape (test_data.shape[0], number of classes)
        """
        log_pred = np.zeros((test_data.shape[0], self.n_classes))

        for i in range(self.n_classes):
            # Here, we will have to use maximum_likelihood_models[i] and priors to fill in
            # each column of log_pred (it's more efficient to do a entire column at a time)
            log_pred[:, i] = self.maximum_likelihood_models[i].loglikelihood(test_data) + np.log(self.priors[i])

        return log_pred

# construct vocab from corpus
def construct_vocab(corpus):
    """
    corpus : numpy array of shape (number of examples in corpus)
    return : list (words in vocab)
    """
    vocab = []
    for sent in corpus:
        words = sent.split()
        for word in words:
            if word not in vocab:
                vocab.append(word)
    return vocab

# construct word frequency matrix
def construct_matrix(vocab, corpus):  
    """
    vocab : list (words in vocab)
    corpus : numpy array of shape (number of examples in corpus)
    return : numpy array of shape (number of examples in corpus, number of features)
    """       
    matrix = np.zeros((corpus.shape[0], len(vocab)))
    for i in range(corpus.shape[0]):
        words = corpus[i].split()
        for word in words:
            if word in vocab:
                matrix[i, vocab.index(word)] += 1
    return matrix

# split training set and validation set in 8:2 and shuffle
def split_data(matrix, y_labels):
    """
    matrix : numpy array of shape (number of examples in corpus, number of features)
    y_labels : numpy array of shape (number of examples in corpus)
    return : 
        train_data : numpy array of shape (number of examples in training set, number of features)
        train_labels : numpy array of shape (number of examples in training set)
        val_data : numpy array of shape (number of examples in validation set, number of features)
        val_labels : numpy array of shape (number of examples in validation set)
    """       
    num_data = matrix.shape[0]
    inds = list(range(num_data))
    random.shuffle(inds)
    train_inds = inds[:int(0.8*num_data)]
    val_inds = inds[int(0.8*num_data):]


    train_data = matrix[train_inds, :]
    train_labels = y_labels[train_inds]
    val_data = matrix[val_inds, :]
    val_labels = y_labels[val_inds]
    return train_data, train_labels, val_data, val_labels

# get accuracy at input data
def get_accuracy(data, labels, classifier):
    """
    data : numpy array of shape (number of examples in corpus, number of features)
    labels : numpy array of shape (number of examples in corpus)
    classifier : object (trained classifier)
    return : float (accuracy for predicting)
    """       
    # Calculate the log-probabilities according to our model
    log_prob = classifier.loglikelihood(data)
    # Predict labels
    classes_pred = log_prob.argmax(1) + 1
    # Calculate the accuracy by comparing the predicted labels with the actual labels
    acc = np.mean(classes_pred == labels)
    return acc

# train procedure
def train(train_data, train_labels, val_data, val_labels):
    """
    train_data : numpy array of shape (number of examples in training set, number of features)
    train_labels : numpy array of shape (number of examples in training set)
    val_data : numpy array of shape (number of examples in validation set, number of features)
    val_labels : numpy array of shape (number of examples in validation set)
    return : object (trained classifier)
    """       
    classes = np.unique(train_labels).shape[0]
    train_classes = ["train" + str(i) for i in range(1, classes + 1)] # corpus type
    for i in range(1, classes + 1):
        train_classes[i - 1] = train_data[train_labels == i, :]
    
    model_class = ["model" + str(i) for i in range(1, classes + 1)]
    # We create a model per class (using maximum likelihood)
    for i in range(1, classes + 1):
        model_class[i - 1] = MultinationalMaxLikelihood()
        model_class[i - 1].train(train_classes[i - 1])
    
    total_num = 0
    for i in range(1, classes + 1):
        total_num += len(train_classes[i - 1])
    priors = [len(train_classes[i - 1]) / (total_num) for i in range(1, classes + 1)]
    
    classifier = BayesClassifier(model_class, priors)
    
    print("The training accuracy is : {:.2f} % ".format(100 * get_accuracy(train_data, train_labels, classifier)))
    print("The validation accuracy is : {:.2f} % ".format(100 * get_accuracy(val_data, val_labels, classifier)))
    return classifier

# get predict result from test data
def get_predict(test_data, classifier):
    """
    test_data : numpy array of shape (number of examples in test set, number of features)
    classifier : object (trained classifier)
    return : list (predict labels for each test example)
    """       
    log_prob = classifier.loglikelihood(test_data)
    classes_pred = log_prob.argmax(1) + 1
    return classes_pred

# write predict labels into csv file
def get_sub(test_Id, test_data, classifier, writepath, category2idx):
    """
    test_Id : list (test examples' id)
    test_data : numpy array of shape (number of examples in test set, number of features)
    classifier : object (trained classifier)
    writepath : str (path for writing csv file)
    category2idx : dictionary (labels mapping)
    return : list (predict labels for each test example) no use here but for further
    """
    y_test = get_predict(test_data, classifier)
    y_test_cat = []
    for label in y_test:
        for value in category2idx:
            if category2idx[value] == label:
                y_test_cat.append(value)
                break
    submisstion = pd.DataFrame({'Id':test_Id,'Category':y_test_cat})
    submisstion.to_csv(writepath, index = False, sep=',')
    print("write to csv finished!")
    return y_test_cat

# main function to run all above function
def main(train_path, test_path, writepath):
    """
    train_path : str (path for train.csv file)
    test_path : str (path for test.csv file)
    writepath : str (path for writing csv file)
    """
    train_Id, train_corpus, train_labels, category2idx = load_data(train_path)
    test_Id, test_corpus = load_data(test_path)
    vocab = construct_vocab(train_corpus)
    train_matrix = construct_matrix(vocab, train_corpus)
    train_data, train_labels, val_data, val_labels = split_data(train_matrix, train_labels)
    classifier = train(train_data, train_labels, val_data, val_labels)
    test_matrix = construct_matrix(vocab, test_corpus)
    y_test = get_sub(test_Id, test_matrix, classifier, writepath, category2idx)
    print("main function finished!")
    
main("./data/train.csv", "./data/test.csv", "./data/sub_NumpyOnly.csv")

