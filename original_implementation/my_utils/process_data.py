#!/usr/bin/env python

import numpy as np
import cPickle as pickle
from collections import defaultdict
import sys, re
from pdb import set_trace


def build_data_cv(train_file, cv=10, clean_string=True, tagField=1, textField=2,userField=None,idField=None):
    """
    Loads data and split into 10 folds.
    :return: sents (with class and split properties), word doc freq, list of labels.
    """
    revs = []
    vocab = defaultdict(int)
    users = {}
    tags = {}
    user = None
    msg_id = None
    i=0
    with open(train_file, "rb") as f:
        for line in f:       
            i+=1
            # if i > 5000: break
            fields = line.strip().split("\t")
            # set_trace()
            text = fields[textField]
            tag = fields[tagField]
            # set_trace()
            if userField is not None:
                user = fields[userField]
            if idField is not None:                
                msg_id = fields[idField]

            if tag not in tags:
                tags[tag] = len(tags)
            if clean_string:
                clean_text = clean_str(text)
            else:
                clean_text = text.lower()
            words = clean_text.split()
            for word in set(words):
                vocab[word] += 1            
            #U2V
            users[user] = 10 #this is hack for the method that loads the embedding matrix
            datum = {"user":user,
                     "y": tags[tag],
                     "text": clean_text,
                     "num_words": len(words),
                     "msg_id" : msg_id,
                     "split": np.random.randint(0, cv)}
            revs.append(datum)
    labels = [0] * len(tags)
    for tag,i in tags.iteritems():
        labels[i] = tag
    return revs, vocab, labels, users


def get_W(word_vecs):
    """
    Get word matrix and word index dict. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    k = len(word_vecs.itervalues().next())
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    for i, word in enumerate(word_vecs):
        W[i+1] = word_vecs[word]
        word_idx_map[word] = i+1
    return W, word_idx_map


def load_word2vec(fname, vocab, binary=True):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        if binary:
            binary_len = np.dtype('float32').itemsize * layer1_size
            for line in xrange(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == ' ':
                        word = ''.join(word)
                        break
                    if ch != '\n':
                        word.append(ch)   
                if word in vocab:
                   word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                else:
                    f.read(binary_len)
        else:                   # text
            for line in f:
                items = line.split()
                word = unicode(items[0], 'utf-8')
                word_vecs[word] = np.array(map(float, items[1:]))
    return word_vecs


def add_unknown_words(word_vecs, vocab, k, min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    :param k: size of embedding vectors.
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)  

def add_unknown_words_zeros(word_vecs, vocab, k, min_df=1):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    :param k: size of embedding vectors.
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.ones(k)  

def tokenize(string, no_lower=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Lower case except when no_lower is Ytur
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if no_lower else string.strip().lower()


def tokenize_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()


def process_data(train_file, clean, w2v_file=None, u2v_file=None,
                 tagField=1, textField=2, userField=3, idField=4, k=300):
    """
    :param k: embeddigs size (300 for GoogleNews)
    """
    np.random.seed(345)         # for replicability
    print "loading data...",
    sents, vocab, labels, users = build_data_cv(train_file, cv=10, clean_string=clean,
                                         tagField=tagField, textField=textField,userField=userField,idField=idField)
    max_l = max(x["num_words"] for x in sents)
    print "data loaded!"
    print "number of sentences: " + str(len(sents))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    if w2v_file:
        print "loading word2vec vectors...",
        w2v = load_word2vec(w2v_file, vocab, w2v_file.endswith('.bin'))
        # get embeddings size:
        k = len(w2v.itervalues().next())
        print "word2vec loaded (%d, %d)" % (len(w2v), k)
        add_unknown_words(w2v, vocab, k)
        W, word_idx_map = get_W(w2v)
    else:
        rand_vecs = {}
        add_unknown_words(rand_vecs, vocab, k)
        W, word_idx_map = get_W(rand_vecs)
    
    if u2v_file:
        u2v = load_word2vec(u2v_file, users, False)        
        j = len(u2v.itervalues().next())
        print "user2vec loaded (%d, %d)" % (len(u2v), j)
        add_unknown_words(u2v, users, j)
        U, user2idx = get_W(u2v)
    else:
        print "Random user vectors"
        rand_vecs = {}
        add_unknown_words(rand_vecs, users, k)
        U, user2idx = get_W(rand_vecs)
    return sents, W, word_idx_map, vocab, labels, max_l, U, user2idx


