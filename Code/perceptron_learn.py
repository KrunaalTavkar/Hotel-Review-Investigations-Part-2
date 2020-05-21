#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:24:27 2020

@author: krunaaltavkar
"""

import os
import sys
import re
import operator
from collections import Counter
import math
import numpy as np
from collections import OrderedDict

class PerceptronClassifier():
    
    def __init__(self):
        # Defining Stop Words and other constants for pattern detection for PerceptronClassifier Object
        self.stop_words = ["the", "and", "a", "i", "to", "was", "in", "of", "we", "is", "hotel",
                           "for", "room", "it", "my", "at", "that", "were", "this", "with",
                           "on", "they", "but", "our", "very", "stay", "you", "t", "be", "would", 
                           "all", "when", "are", "service", "one", "rooms", "stayed", "up", "s", 
                           "if", "like", "night", "just", "which", "back", "check", "first", "view", 
                           "about", "after", "location", "bed", "even", "by", "could", "here", "what", 
                           "other", "next", "because", "friendly", "go", "your", "has", "any", 
                           "price", "right", "people", "make", "into", "wife", "checked", "don", 
                           "seemed", "family", "extremely", "money", "upon", "loved", "able", 
                           "see", "highly", "worth", "kinckerbocker", "part", "size", "double", 
                           "choice", "cost", "charged", "point", "side", "employees", ""]
        
        self.pos_pattern = 'positive'
        self.neg_pattern = 'negative'
        self.tru_pattern = 'truthful'
        self.dec_pattern = 'deceptive'
        self.train_set_pattern = r"(LICENSE|README|DS_Store)"
        self.cleaned_train_data = []
        self.cleaned_dev_data = []
        self.list_of_unique_words_in_training_corpus = []
    
    def get_training_documents(self, input_path):
        # Paths for data are read from command line
        train_file = input_path
        train_set = []
        for root, dirnames, files in os.walk(train_file):
            for file in files:
                file_str = root + '/' + file
                if not re.search(self.train_set_pattern, file_str):
                    train_set.append(file_str)
        
        return train_set
    
    def get_training_tokens_and_labels(self, train_set):
        cleaned_train_data = []
        train_sentiments = []
        train_tru_decep = []
        vocabulary = []
        pos_count = 0
        neg_count = 0
        tru_count = 0
        decep_count = 0
        for i in range(len(train_set)):
            f = open(train_set[i], 'r')
            data = f.read()
            data = data.lower()
            data = re.sub('[^a-z\s]+', " ", data)
            data = re.sub('(\s+)', " ", data)
            cleaned_train_data.append(data)
            if re.search(self.pos_pattern, train_set[i]):
                
                polarity = 1
            else:
                
                polarity = -1
            if re.search(self.tru_pattern, train_set[i]):
                
                truthful = 1
            else:
                
                truthful = -1
                
            train_sentiments.append(polarity)
            train_tru_decep.append(truthful)

        for i in range(len(cleaned_train_data)):
            all_words = [word for word in re.split("\s+", cleaned_train_data[i]) if word not in self.stop_words]
            for word in all_words:
                vocabulary.append(word)
        
        vocabulary_counter = Counter(vocabulary)
        remove_words = [word for word in vocabulary_counter if vocabulary_counter[word] < 2]
        self.list_of_unique_words_in_training_corpus = list(vocabulary_counter.keys())
        self.list_of_unique_words_in_training_corpus = [word for word in self.list_of_unique_words_in_training_corpus if word not in remove_words]
        self.list_of_unique_words_in_training_corpus = sorted(self.list_of_unique_words_in_training_corpus)
        #print(len(self.list_of_unique_words_in_training_corpus))
        
        for i in range(len(cleaned_train_data)):
            row_vector = OrderedDict()
            row_vector = OrderedDict().fromkeys(self.list_of_unique_words_in_training_corpus, 0)
            all_words = [word for word in re.split("\s+", cleaned_train_data[i]) if word not in self.stop_words and word in self.list_of_unique_words_in_training_corpus]
            for word in all_words:
                row_vector[word] += 1
            for word in list(row_vector.values()):
                self.cleaned_train_data.append(word)
       # print(row_vector)
#        print(len(self.cleaned_train_data))
        vectorised_train_data = np.asarray(self.cleaned_train_data)
        vectorised_train_data = vectorised_train_data.reshape(len(cleaned_train_data), len(self.list_of_unique_words_in_training_corpus))
        
        
        return vectorised_train_data, train_sentiments, train_tru_decep

    
    def vanilla_fit(self, vectorised_train_data, train_sentiments, max_iterations=100):
        N , D = vectorised_train_data.shape
        X = vectorised_train_data
#        print(X.shape)
        y = np.asarray(train_sentiments)
        weights = np.zeros(D)
        bias = 0
#        print(y.shape)
        for iterations in range(max_iterations):
            gradient = 0
            error = 0
            Z = y*(np.dot(X, weights) + bias)
            preds = np.where(Z<=0, 1, 0)
            gradient = np.dot(y*preds, X)
            error = np.sum(y*preds)
            weights += gradient
            bias += error
        
        return weights, bias
    
    def averaged_fit(self, vectorised_train_data, train_sentiments, max_iterations=100):
        N , D = vectorised_train_data.shape
        X = vectorised_train_data
#        print(X.shape)
        y = np.asarray(train_sentiments)
        weights = np.zeros(D)
        bias = 0
        cached_weights = np.zeros(D)
        beta = 0
        counter = 1
#        print(y.shape)
        for iterations in range(max_iterations):
            gradient = 0
            error = 0
            Z = y*(np.dot(X, weights) + bias)
            preds = np.where(Z<=0, 1, 0)
            gradient = np.dot(y*preds, X)
            error = np.sum(y*preds)
            weights += gradient
            bias += error
            cached_weights += gradient*counter
            beta += error*counter
            counter += 1
        
        weights -= (cached_weights/counter)
        bias -= (beta/counter)
        
        return weights, bias
    
    
    def predict(self, vectorised_dev_data, weights, bias):
        N, D = vectorised_dev_data.shape
        
        Z = np.dot(vectorised_dev_data, weights) + bias
        pred_dev_sentiments = np.where(Z>0, 1, -1)
            
        return pred_dev_sentiments
    
    def generate_vanilla_model(self, sentiment_weights, sentiment_bias, tru_decep_weights, tru_decep_bias):

        write_rows = []
        row_headers = ['sentiment_weights', 'sentiment_bias', 'tru_decep_weights', 'tru_decep_bias', 'list_of_unique_words_in_training_corpus']
        row_values = [sentiment_weights, sentiment_bias, tru_decep_weights, tru_decep_bias, self.list_of_unique_words_in_training_corpus]
        
        for word in range(len(row_headers)):
            
            row = [row_headers[word], row_values[word]]
            write_rows.append(row)
#        print(write_rows)
        with open('vanillamodel.txt', 'w') as f:
#            
            for index in range(len(write_rows)):
                my_str = ""
                for i in range(len(write_rows[index])):
#                    print(write_rows[index][i])
                    if index == 0 or index == 2 or index == 4:
                        if i == 0:
                            my_str += str(write_rows[index][i]) + "\t"
                        else:
                            for weight in write_rows[index][i]:
                                my_str += str(weight) + "\t"
                    else:
                        my_str += str(write_rows[index][i]) + "\t"
                my_str += "\n"
                f.writelines(my_str)
                
    def generate_averaged_model(self, sentiment_weights, sentiment_bias, tru_decep_weights, tru_decep_bias):

        write_rows = []
        row_headers = ['sentiment_weights', 'sentiment_bias', 'tru_decep_weights', 'tru_decep_bias', 'list_of_unique_words_in_training_corpus']
        row_values = [sentiment_weights, sentiment_bias, tru_decep_weights, tru_decep_bias, self.list_of_unique_words_in_training_corpus]
        
        for word in range(len(row_headers)):
            
            row = [row_headers[word], row_values[word]]
            write_rows.append(row)
#        print(write_rows)
        with open('averagedmodel.txt', 'w') as f:
#            
            for index in range(len(write_rows)):
                my_str = ""
                for i in range(len(write_rows[index])):
#                    print(write_rows[index][i])
                    if index == 0 or index == 2 or index == 4:
                        if i == 0:
                            my_str += str(write_rows[index][i]) + "\t"
                        else:
                            for weight in write_rows[index][i]:
                                my_str += str(weight) + "\t"
                    else:
                        my_str += str(write_rows[index][i]) + "\t"
                my_str += "\n"
                f.writelines(my_str)

if __name__== "__main__":
    input_path = sys.argv[1]
    nb_object = PerceptronClassifier()
    train_set = nb_object.get_training_documents(input_path)
    vectorised_train_data, train_sentiments, train_tru_decep = nb_object.get_training_tokens_and_labels(train_set)
    sentiment_weights, sentiment_bias = nb_object.vanilla_fit(vectorised_train_data, train_sentiments)
    tru_decep_weights, tru_decep_bias = nb_object.vanilla_fit(vectorised_train_data, train_tru_decep)
    nb_object.generate_vanilla_model(sentiment_weights, sentiment_bias, tru_decep_weights, tru_decep_bias)
  
    sentiment_weights, sentiment_bias = nb_object.averaged_fit(vectorised_train_data, train_sentiments)
    tru_decep_weights, tru_decep_bias = nb_object.averaged_fit(vectorised_train_data, train_tru_decep)
    nb_object.generate_averaged_model(sentiment_weights, sentiment_bias, tru_decep_weights, tru_decep_bias)
    