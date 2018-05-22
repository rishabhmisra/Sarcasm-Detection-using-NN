import argparse
import csv
import codecs
from pdb import set_trace
import os
import sys
import re
sys.path.append("code")
from my_utils import embeddings as emb_utils
import my_utils as ut

FOLDS =  "DATA/folds/"
def build_folds(msg_ids):
    if not os.path.isdir(FOLDS):
        os.mkdir(FOLDS)
    kf = ut.kfolds(10, len(msg_ids),val_set=True,shuffle=True)
    for i, fold in enumerate(kf):
        fold_data = {"train": [ int(msg_ids[x]) for x in fold[0] ], 
                     "test" : [ int(msg_ids[x]) for x in fold[1] ],
                     "val"  : [ int(msg_ids[x]) for x in fold[2] ] }
        with open(FOLDS + 'fold_%d.csv' % i, 'wb') as f: 
            w = csv.DictWriter(f, fold_data.keys())
            w.writeheader()
            w.writerow(fold_data)


def get_parser():
    parser = argparse.ArgumentParser(description="Preprocess Bamman and Smith sarcasm dataset")
    parser.add_argument('-input', type=str, required=True, 
                        help='dataset')        
    parser.add_argument('-out_txt', type=str, required=True, 
                        help='path for the preprocessed files')
    parser.add_argument('-word_vectors', type=str, required=True, 
                        help='path to WORD embeddings')
    parser.add_argument('-out_vectors', type=str, required=True, 
                        help='path to output the subset of the embeddings for the words that occur in the training data')
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args   = parser.parse_args()    
    idz = []
    print "Preprocess Data"
    with codecs.open(args.out_txt,"w","utf-8") as fod:
        with codecs.open(args.input,"r","utf-8") as fid:
            msgs = []                        
            for line in fid:                
                clean_line = re.sub('[\n\r\'\"]', '', line)
                clean_line = clean_line.replace("#sarcasm", "").replace("#sarcastic", "")                
                st = clean_line.split("\t")
                if len(st) != 4: set_trace()
                tweet_id, user, label, m = st
                idz.append(int(tweet_id)) 
                m = ut.preprocess(m, sep_emoji=True)
                fod.write(u"%s\t%s\t%s\t%s\n" % (tweet_id,user,label,m))
                msgs.append(m)
    #compute word index
    wrd2idx = ut.word_2_idx(msgs)    
    print "Load Word Embeddings"
    emb_utils.save_embeddings_txt(args.word_vectors, args.out_vectors, wrd2idx)
    # pre-compute the crossvalidation folds so that different models 
    # can be compared on the same data splits
    build_folds(idz)




