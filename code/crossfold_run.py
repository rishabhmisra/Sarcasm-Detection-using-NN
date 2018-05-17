#!/usr/bin/env python

"""
Training a convolutional network for sentence classification,
as described in paper:
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf
"""
import ast
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import csv
import cPickle as pickle
import numpy as np
import theano
import sys
import argparse
import warnings
from pdb import set_trace
import itertools
warnings.filterwarnings("ignore")   
import pprint
sys.path.append(".")
from convnets import *
from process_data import process_data
csv.field_size_limit(sys.maxsize)
import time
import sys

def get_idx_from_sent(sent, word_index, max_l, pad):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    Drop words non in word_index. Attardi.
    :param max_l: max sentence length
    :param pad: pad length
    """
    x = [0] * pad                # left padding
    words = sent.split()[:max_l] # truncate words from test set
    for word in words:
        if word in word_index: # FIXME: skips unknown words
            x.append(word_index[word])
    while len(x) < max_l + 2 * pad: # right padding
        x.append(0)
    return x


def make_idx_data_cv(revs, word_index, user_index, cv, max_l, pad):
    """
    Transforms sentences into a 2-d matrix and splits them into
    train and test according to cv.
    :param cv: cross-validation step
    :param max_l: max sentence length
    :param pad: pad length
    """
    
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_index, max_l, pad)
        sent.append(user_idx[rev["user"]])        
        sent.append(rev["y"])
        if rev["split"] == cv:
            test.append(sent)        
        else:  
            train.append(sent)   
    train = np.array(train, dtype="int32")
    test = np.array(test, dtype="int32")
    return train, test


def make_idx_data_from_ids(revs, word_index, user_index, max_l, pad, msg_ids):
    """
    Transforms sentences into a 2-d matrix and splits them into
    train and test according to cv.
    :param cv: cross-validation step
    :param max_l: max sentence length
    :param pad: pad length
    """
    
    data = []
    for rev in revs:        
        if int(rev["msg_id"]) in msg_ids:
            sent = get_idx_from_sent(rev["text"], word_index, max_l, pad)
            sent.append(user_idx[rev["user"]])        
            sent.append(rev["y"])        
            data.append(sent)   
    data = np.array(data, dtype="int32")    
    return data

def make_idx_data(revs, word_index, user_index, max_l, pad):
    """
    Transforms sentences into a 2-d matrix     
    :param max_l: max sentence length
    :param pad: pad length
    """    
    train = []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_index, max_l, pad)
        sent.append(user_idx[rev["user"]])        
        sent.append(rev["y"])
        train.append(sent)   
    train = np.array(train, dtype="int32")    
    return train

def read_corpus(filename, word_index, max_l, pad=2, clean_string=False,
                textField=3):
    test = []
    with open(filename) as f:
        for line in f:
            fields = line.strip().split("\t")
            text = fields[textField]
            if clean_string:
                text_clean = clean_str(text)
            else:
                text_clean = text.lower()
            sent = get_idx_from_sent(text_clean, word_index, max_l, pad)
            #sent.append(0)      # unknown y
            test.append(sent)
    return np.array(test, dtype="int32")

def search_architecture(W, user_idx, max_l, params):

    width          = params["filter_width"]
    hidden_units   = params["hidden units"] 
    dropout_rate   = params["dropout rate"]
    batch_size     = params["batch size"]
    lr_decay       = params["adadelta decay"]
    conv_non_linear = params["conv_non_linear"]
    non_static     = params["non static"] 
    sqr_norm_lim   = params["sqr_norm_lim"]
    shuffle_batch  = params["shuffle batch"]    
    epochs         = params["epochs"]    
                  
    
    # ensure replicability
    np.random.seed(3435)
    rng = np.random.RandomState(3435)
    # perform 10-fold cross-validation
    n_filters    = [3,5,7]
    filter_sizes = [1,3,5,7,9]
    n_hiddens     = [50,100,200,400,800] 
    confs = list(itertools.product(filter_sizes,n_hiddens,n_filters))
    rng.shuffle(confs)
    # set_trace()
    # for i,c in enumerate(confs): print i,c
    try:
        with open("DATA/pkl/arch_results_all.pkl","r") as fid:
                config_results = pickle.load(fid)
    except:
        print "could not open previous config results"
        config_results = {}    

    for conf in confs[:70]:

        if conf in config_results:
            print "Already evaluated this conf"
            continue
        print "\n\nEVALUATING CONF: ", repr(conf)
        time.sleep(500)
        print "\n\n"
        filter_size = conf[0]
        n_hiddens   = conf[1]
        n_filter    = conf[2]

        hidden_units[0] = n_hiddens
        filter_hs = [filter_size] * n_filter
        pad = max(filter_hs) - 1
        # max_l = 56 # DEBUG: max(x["num_words"] for x in sents)
        height = max_l + 2 * pad # padding on both sides
        results = []
        # set_trace()
        for i in range(0, 10):
            # test = [padded(x) for x in sents if x[split] == i]
            # train is rest
            train_set, test_set = make_idx_data_cv(sents, word_index, user_idx, i, max_l, pad)        
            cnn = ConvNet(W, None, height, width,
                          filter_hs=filter_hs,
                          conv_non_linear=conv_non_linear,
                          hidden_units=hidden_units,
                          batch_size=batch_size,
                          non_static=non_static,
                          dropout_rates=[dropout_rate])

            perf = cnn.evaluate(train_set, test_set,
                             lr_decay=lr_decay,
                             shuffle_batch=shuffle_batch, 
                             epochs=epochs,
                             sqr_norm_lim=sqr_norm_lim)
            print "cv: %d, perf: %f" % (i, perf)
            results.append(perf)  
            # save model
            if i == 0 or perf > max(results):
                with open(model, "wb") as mfile:
                    cnn.save(mfile)
                    pickle.dump((word_index, labels, max_l, pad), mfile)
        
        config_results[conf] = (results, round(np.mean(results),4))
        print "Avg. accuracy: %.4f" % np.mean(results)
        with open("DATA/pkl/arch_results_all.pkl","w") as fid:
            pickle.dump(config_results,fid,-1)



def test_fold(W, U, user_idx, max_l, params, folds_file):

    width          = params["filter_width"]    
    batch_size     = params["batch size"]
    lr_decay       = params["adadelta decay"]
    conv_non_linear = params["conv_non_linear"]
    non_static     = params["non static"] 
    sqr_norm_lim   = params["sqr_norm_lim"]
    shuffle_batch  = params["shuffle batch"]    
    epochs         = params["epochs"]                      
    
    # ensure replicability
    np.random.seed(3435)
    rng = np.random.RandomState(3435)
    # perform 10-fold cross-validation
    # n_filters    = [3,5,7]
    # filter_sizes = [1,3,5,7,9]
    # n_hiddens     = [50,100,200,400,800] 
    # confs = list(itertools.product(filter_sizes,n_hiddens,n_filters))
    # rng.shuffle(confs)
    filters    = [(1,3,5),(2,4,6),(3,5,7),(4,6,8),(5,7,9)]
    n_feat_maps = [100,200,400,600]
    drop_outs  = [0.0, 0.1]
    n_hiddens  = [100,75,50,25]
    confs = list(itertools.product(filters,n_feat_maps,drop_outs,n_hiddens))
    rng.shuffle(confs)
    print len(confs)
    # set_trace()
    # for i,c in enumerate(confs): print i,c    

    train_ids, test_ids, val_ids = get_folds(args.folds)

    config_results = {}

    for conf in confs[:int(len(confs)*0.4)]:        
        print "\n\nEVALUATING CONF: ", repr(conf)
        #time.sleep(500)
        print "\n\n"
        # filter_size = conf[0]
        
        # n_filter    = conf[2]

        filter_hs       = conf[0]
        hidden_units[0] = conf[1]
        hidden_units[1] = conf[3]
        dropout_rate    = conf[2] 
        
        filter_hs       = (1,3,5)
        hidden_units[0] = 200
        hidden_units[1] = 100
        dropout_rate    = 0.1

        # hidden_units[0] = n_hiddens
        # filter_hs = [filter_size] * n_filter
        pad = max(filter_hs) - 1
        height = max_l + 2 * pad # padding on both sides
        results = []
        train_set = make_idx_data_from_ids(sents, word_index, user_idx, max_l, pad, train_ids)      
        val_set = make_idx_data_from_ids(sents, word_index, user_idx, max_l, pad, val_ids)    
        
        cnn = ConvNet(W, U, height, width,
                      filter_hs=filter_hs,
                      conv_non_linear=conv_non_linear,
                      hidden_units=hidden_units,
                      batch_size=batch_size,
                      non_static=non_static,
                      dropout_rates=[dropout_rate],
                      activations=[ReLU,Iden])

        perf = cnn.evaluate(train_set, val_set,
                         lr_decay=lr_decay,
                         shuffle_batch=shuffle_batch, 
                         epochs=epochs,
                         sqr_norm_lim=sqr_norm_lim)
            
        # # save model
        # if i == 0 or perf > max(results):
        #     with open(model, "wb") as mfile:
        #         cnn.save(mfile)
        #         pickle.dump((word_index, labels, max_l, pad), mfile)
        
        config_results[conf] = round(perf,3)
        print "Accuracy (For Loop): %.4f" % perf
        break
    
    top_conf = sorted(config_results.items(), key=lambda x:x[1])[-1]
    print "The top conf is:", top_conf
#     conf        = top_conf[0]
#     filter_hs       = conf[0]
#     hidden_units[0] = conf[1]
#     hidden_units[1] = conf[3]
#     dropout_rate    = conf[2] 
#     # hidden_units[0] = n_hiddens
#     # filter_hs = [filter_size] * n_filter
#     pad = max(filter_hs) - 1
#     height = max_l + 2 * pad # padding on both sides

#     train_set = make_idx_data_from_ids(sents, word_index, user_idx, max_l, pad, train_ids)      
#     val_set = make_idx_data_from_ids(sents, word_index, user_idx, max_l, pad, val_ids)    
#     test_set  = make_idx_data_from_ids(sents, word_index, user_idx, max_l, pad, test_ids)   
    
#     all_set = np.concatenate((train_set,val_set),0)

#     cnn = ConvNet(W, U, height, width,
#                   filter_hs=filter_hs,
#                   conv_non_linear=conv_non_linear,
#                   hidden_units=hidden_units,
#                   batch_size=batch_size,
#                   non_static=non_static,
#                   dropout_rates=[dropout_rate],
#                   activations=[ReLU,Iden])

#     perf = cnn.evaluate(all_set, test_set,
#                      lr_decay=lr_decay,
#                      shuffle_batch=shuffle_batch, 
#                      epochs=epochs,
#                      sqr_norm_lim=sqr_norm_lim)
        
    # # save model
    # if i == 0 or perf > max(results):
    #     with open(model, "wb") as mfile:
    #         cnn.save(mfile)
    #         pickle.dump((word_index, labels, max_l, pad), mfile)
    
    #config_results[conf] = (results, round(perf))
    #print "Accuracy final: %.4f" % perf
        
    
def get_parser():

    parser = argparse.ArgumentParser(description="CNN sentence classifier.")
    
    parser.add_argument('model', type=str, default='mr',
                        help='model file (default %(default)s)')
    parser.add_argument('input', type=str,
                        help='train/test file in SemEval twitter format')
    parser.add_argument('-vectors', type=str,
                        help='word2vec embeddings file (random values if missing)')
    parser.add_argument('-user_vectors', type=str, default=None,
                        help='word2vec embeddings file (random values if missing)')
    parser.add_argument('-epochs', type=int, default=25,
                        help='training iterations (default %(default)s)')
    parser.add_argument('-tagField', type=int, default=2,
                        help='label field in files (default %(default)s)')
    parser.add_argument('-textField', type=int, default=3,
                        help='text field in files (default %(default)s)')
    parser.add_argument('-userField', type=int, default=1,
                        help='user field in files (default %(default)s)')
    parser.add_argument('-idField', type=int, default=0,
                        help='user field in files (default %(default)s)')
    parser.add_argument('-folds', type=str, 
                        help='folds files')
    parser.add_argument('-context', type=int, default=0, 
                        help='context')

    return parser

def get_folds(folds_file):

    with open(folds_file ) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_id = ast.literal_eval(row['train'])
            test_id = ast.literal_eval(row['test'])
            val_id = ast.literal_eval(row['val'])
    
    return train_id, test_id, val_id

if __name__=="__main__":

    parser = get_parser()
    args = parser.parse_args()    
    

    # training
    sents, W, word_index, vocab, labels, max_l, U, user_idx = process_data(args.input, False, args.vectors, args.user_vectors, args.tagField, args.textField, args.userField, idField=args.idField)    
    # set_trace()
    model = args.model
    # if args.static:
    #     print "model architecture: CNN-static"
    #     non_static = False
    # else:
    #     print "model architecture: CNN-non-static"
    #     non_static = True
    non_static = True
    if args.vectors:
        print "using: word2vec vectors"
    else:
        print "using: random vectors"

    classes = set(x["y"] for x in sents)
    width = W.shape[1]
    conv_non_linear = "relu"
    batch_size = 32
    dropout_rate = 0
    sqr_norm_lim = 11
    shuffle_batch = True
    lr_decay = 0.95
    hidden_units = [100, 50, len(classes)]
    # filter_h determines padding, hence it depends on largest filter size.
    # layer_sizes = [hidden_units[0]*len(filter_hs)] + hidden_units[1:]
    if args.context==0:
        U = None
    # set_trace()
    parameters = {"filter_width": width,
                  "filters": None,
                  "hidden units": hidden_units,                  
                  "dropout rate": dropout_rate,
                  "batch size": batch_size,
                  "adadelta decay": lr_decay,
                  "conv_non_linear": conv_non_linear,
                  "non static": non_static,
                  "sqr_norm_lim": sqr_norm_lim,
                  "shuffle batch": shuffle_batch,
                  # "epochs":args.epochs}
                  "epochs":300}

    # for param,val in parameters.items():
    #     # print "%s: %s" % (param, ",".join(str(x) for x in val))
    #     print "%s: %s" % (param, repr(val))
    
    test_fold(W, U, user_idx, max_l, parameters,args.folds)



    
