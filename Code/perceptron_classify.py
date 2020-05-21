#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:18:56 2020

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
        # Defining Stop Words and other constants for pattern detection for NBClassifier Object
        self.stop_words = ["the", "and", "a", "i", "to", "was", "in", "of", "we", "is", "hotel",
                           "for", "room", "it", "my", "at", "that", "were", "this", "with",
                           "on", "they", "but", "our", "very", "stay", "you", "t", "be", "would", 
                           "all", "when", "are", "service", "one", "rooms", "stayed", "up", "s", 
                           "if", "like", "night", "just", "which", "back", "check", "first", "view", 
                           "about", "after", "location", "bed", "even", "by", "could", "here", "what", 
                           "other", "next", "because", "friendly", "go", "your", "has", "any", 
                           "price", "right", "people", "make", "into", "wife", "checked", "don", 
                           "seemed", "family", "extremely", "money", "upon", "enough", "loved", "able", 
                           "see", "highly", "worth", "kinckerbocker", "part", "size", "double", 
                           "choice", "cost", "charged", "point", "side", "employees", ""]
        
        self.sentiment_weights = []
        self.sentiment_bias = 0
        self.tru_decep_weights = []
        self.tru_decep_bias = 0
        self.list_of_unique_words_in_training_corpus = []
        self.cleaned_test_data = []
        self.test_set_pattern = r"(LICENSE|README|DS_Store)"
        self.pos_pattern = 'positive'
        self.neg_pattern = 'negative'
        self.tru_pattern = 'truthful'
        self.dec_pattern = 'deceptive'
    
    def get_model(self, model_path):
        with open(model_path, 'r') as document:
            for line in document:
                line = line.split()
                if not line:  # empty line?
                    continue
                if line[0] == 'sentiment_weights':
                    line[1:] = [float(i) for i in line[1:]]
                    self.sentiment_weights = line[1:]
                elif line[0] == 'tru_decep_weights':
                    line[1:] = [float(i) for i in line[1:]]
                    self.tru_decep_weights = line[1:]
                elif line[0] == 'sentiment_bias':
                    self.sentiment_bias = float(line[1])
                elif line[0] == 'tru_decep_bias':
                    self.tru_decep_bias = float(line[1])
                elif line[0] == 'list_of_unique_words_in_training_corpus':
                    self.list_of_unique_words_in_training_corpus = list(line[1:])
        
        self.sentiment_weights = np.asarray(self.sentiment_weights)
        self.tru_decep_weights = np.asarray(self.tru_decep_weights)
        
        
        return self.sentiment_weights, self.sentiment_bias, self.tru_decep_weights, self.tru_decep_bias 
        
    def get_test_documents(self, input_path):
        test_set = []
        for root, dirnames, files in os.walk(input_path):
            for file in files:
                file_str = root + '/' + file
                if not re.search(self.test_set_pattern, file_str):
                    test_set.append(file_str)
        
        return test_set
    
    def get_test_tokens(self, test_set):
        cleaned_test_data = []
        test_sentiments = []
        test_tru_decep = []
        for i in range(len(test_set)):
            f = open(test_set[i], 'r')
            data = f.read()
            data = data.lower()
            data = re.sub('[^a-z\s]+', " ", data)
            data = re.sub('(\s+)', " ", data)
            cleaned_test_data.append(data)
            if re.search(self.pos_pattern, test_set[i]):
                polarity = 1
            else:
                polarity = -1
            if re.search(self.tru_pattern, test_set[i]):
                truthful = 1
            else:
                truthful = -1
            test_sentiments.append(polarity)
            test_tru_decep.append(truthful)
        
        for i in range(len(cleaned_test_data)):
            row_vector = OrderedDict()
            row_vector = OrderedDict.fromkeys(self.list_of_unique_words_in_training_corpus, 0)
            all_words = [word for word in re.split("\s+", cleaned_test_data[i]) if word not in self.stop_words]
            for word in all_words:
                if word in self.list_of_unique_words_in_training_corpus:
                    row_vector[word] += 1
            for word in list(row_vector.values()):
                self.cleaned_test_data.append(word)
        
#        print(len(self.cleaned_train_data))
        vectorised_test_data = np.asarray(self.cleaned_test_data)
        vectorised_test_data = vectorised_test_data.reshape(len(cleaned_test_data), len(self.list_of_unique_words_in_training_corpus))
#        np.random.seed(0)
#        np.random.shuffle(vectorised_test_data)
        
        return vectorised_test_data, test_sentiments, test_tru_decep

    def predict(self, vectorised_test_data, weights, bias):
        N, D = vectorised_test_data.shape
        
        Z = np.dot(vectorised_test_data, weights) + bias
        pred_test_sentiments = np.where(Z>0, 1, -1)
#        print(pred_test_sentiments)
        return pred_test_sentiments
    
    def generate_output_contents(self, pred_test_sentiments, pred_test_tru_decep, test_set):
        output_data = []
        counter_val = 0
        labels = [(x,y) for x, y in zip(pred_test_sentiments, pred_test_tru_decep)]
#        print(labels)
        for entry in labels:
            output_row = []
            if entry == (1, 1):
                output_row.append('truthful')
                output_row.append('positive')
            elif entry == (-1, 1):
                output_row.append('truthful')
                output_row.append('negative')
            elif entry == (-1, -1):
                output_row.append('deceptive')
                output_row.append('negative')
            else:
                output_row.append('deceptive')
                output_row.append('positive')
            output_row.append(test_set[counter_val])
            counter_val += 1
            output_data.append(output_row)
        
        return output_data

    def generate_output(self, output_data):
        with open('percepoutput.txt', 'w') as f:
            for row in output_data:
                my_str = ""
                for word in row:
                    my_str += str(word) + "\t"
                my_str += "\n"
                f.writelines(my_str)

if __name__== "__main__":
    model_path = sys.argv[1]
    input_path = sys.argv[2]
    nb_object = PerceptronClassifier()
    t_sentiment_weights, t_sentiment_bias, t_tru_decep_weights, t_tru_decep_bias = nb_object.get_model(model_path)
    test_set = nb_object.get_test_documents(input_path)
    vectorised_test_data, test_sentiments, test_tru_decep = nb_object.get_test_tokens(test_set)
    pred_test_sentiments = nb_object.predict(vectorised_test_data, t_sentiment_weights, t_sentiment_bias)
    pred_test_tru_decep = nb_object.predict(vectorised_test_data, t_tru_decep_weights, t_tru_decep_bias)
    output_data = nb_object.generate_output_contents(pred_test_sentiments, pred_test_tru_decep, test_set)
    nb_object.generate_output(output_data)