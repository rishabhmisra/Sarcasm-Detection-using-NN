import ast
import csv
import sys
import numpy as np
from torch.utils.data.dataset import Dataset
from collections import defaultdict
from pdb import set_trace as brk
import torch
import pandas as pd

class SemEvalDataset(Dataset):
    """SemEval dataset."""

    def __init__(self, csv_file, folds_file, word_embedding_file, user_embedding_file, set_type, pad, w2v=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with tweets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv.field_size_limit(sys.maxsize)
        self.csv = pd.read_csv(csv_file)
        
        w2v = self.load_word2vec(word_embedding_file, vocab, word_embedding_file.endswith('.bin'))
        # get embeddings size:
        k = len(w2v.itervalues().next())
        print "word2vec loaded (%d, %d)" % (len(w2v), k)
        #self.add_unknown_words(w2v, vocab, k)
        self.w2v = w2v
        self.max_l = max_l
        self.pad = pad
        self.transform = transform


    def load_word2vec(self,fname, vocab, binary=True):
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


    def add_unknown_words(self,word_vecs, vocab, k, min_df=1):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        :param k: size of embedding vectors.
        """
        for word in vocab:
            if word not in word_vecs and vocab[word] >= min_df:
                word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


    def __len__(self):
        print "LENGTHHH...", len(self.csv)
        return len(self.csv)

    def __getitem__(self, idx):
        label = self.csv.iloc[idx, 1]
        sent = self.csv.iloc[idx, 2]
       
        k = self.w2v.itervalues().next()
        x = [[0] * k for i in range(self.pad)] 
        words = sent.split()[:self.max_l] # truncate words from test set
        for word in words:
            if word in self.w2v: # FIXME: skips unknown words
                x.append(self.w2v[word])
            else:
                x.append(np.random.uniform(-0.25, 0.25, k))
        while len(x) < self.max_l + 2 * self.pad : # right padding
            x.append([0] * k)

        return  torch.from_numpy(np.array(x)).unsqueeze(0), torch.zeros(400), label, sent