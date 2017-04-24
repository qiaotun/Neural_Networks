#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:49:07 2017

@author: luohaosheng
@function: read text files and count the # of words in each line for NLP; order of words are not considered; capable of handing large dataset
@pre-requisite: nltk and sentiment 140 data 
"""

import os
path = ‘your_own_path’
os.chdir(path)

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import numpy as np
import random
import pandas as pd
from collections import Counter

lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

def init_process(fin,fout, buffer_size=5000000, encode='latin-1'):
    outfile = open(fout,'w', encoding=encode)
    with open(fin, 'r', buffering=buffer_size, encoding=encode) as f:
        try:
            for line in f:
                line = line.replace('"','')
                initial_polarity = line.split(',')[0]
                if initial_polarity == '0':
                    initial_polarity = [1,0,0]
                elif initial_polarity == '2':
                    initial_polarity = [0,1,0]                
                elif initial_polarity == '4':
                    initial_polarity = [0,0,1]

                tweet = line.split(',')[-1]
                outline = str(initial_polarity)+':::'+tweet
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()
    
init_process('training.1600000.processed.noemoticon.csv','train_set.csv')
init_process('testdata.manual.20090614.csv','test_set.csv')


def create_lexicon(fin, buffer_size=5000000, encode='latin-1'):
    # languages available in nltk with a stopword library
    languages = ['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian',
            'italian', 'norwegian', 'portuguese', 'russian', 'spanish', 'swedish', 'turkish']

    with open(fin, 'r', buffering=buffer_size, encoding=encode) as f:
        try:        
            content = str.join(':::',[line.strip() for line in f])
            content = content.lower().split(':::')[1::2]
            content = ' '+str.join(' ',[l.strip() for l in content])
            #this tokenizer does not remove punctuation
            words = word_tokenize(content)
            unique_words = list(set(words))
            
        except Exception as e:
            print(str(e))
        
        for l in languages:
            cachedStopWords = stopwords.words(l)
            unique_words = [w for w in unique_words if w not in cachedStopWords]
            print('Start removing ', l, 'stop words.')
            
        # create dictionary of word counts of all lines
        w_counts = Counter(words) 
        words = []
        count = 0
        for w in w_counts:
            count += 1
            # thresholds are hard coded, can be improved
            if 150 < w_counts[w] < 100000:
                # generate a list of words that may be important
                words.append(w) 
        
        lexicon = list(set(unique_words).intersection(set(words)))
    print('Lexicon size is: ', len(lexicon))
    with open('lexicon.pickle','wb') as fo:
        pickle.dump(lexicon, fo)

create_lexicon('train_set.csv')
    
def convert_to_mat(fin,fout,lexicon_pickle, buffer_size=5000000, encode='latin-1'):
    with open(lexicon_pickle,'rb') as f:
        lexicon = pickle.load(f)
        
    with open(fin, buffering=buffer_size, encoding=encode) as f:
        counter = 0
        features_label = []
        for line in f:
            label = eval(line.split(':::')[0])
            
            tweet = line.split(':::')[1] #how to deal with neutral?
            current_words = word_tokenize(tweet.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            
            features = [0]*len(lexicon)
            for word in current_words:
                if word in lexicon:
                    index_value = lexicon.index(word)
                    features[index_value] += 1
            
            features_label_l = [features, label]
            features_label.append(features_label_l)
            counter += 1
            
    print("Rows of test dataset:", counter)
    # output is a list of two lists: features and labels
    with open(fout,'wb') as fo:
        pickle.dump(features_label, fo)

convert_to_mat('test_set.csv','test_set_processed.pickle','lexicon.pickle')


def shuffle_train_data(fin, fout, encode = 'latin-1'):
    df = pd.read_csv(fin, error_bad_lines=False, encoding=encode)
    df = df.iloc[np.random.permutation(len(df))]
    print('First 3 rows of the shuffled data:')
    print(df.head(3))
    df.to_csv(fout, index=False)

shuffle_train_data('train_set.csv','train_set_shuffled.csv')
