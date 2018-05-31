import ast
import csv
import sys
import numpy as np
from torch.utils.data.dataset import Dataset
from collections import defaultdict
from pdb import set_trace as brk
import torch
import pandas as pd

class HeadlineDataset(Dataset):
    """SemEval dataset."""

    def __init__(self, csv_file, word_embedding_file, pad, whole_data=None, word_idx=None, pretrained_embs=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with tweets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv.field_size_limit(sys.maxsize)
        self.csv = pd.read_csv(csv_file, sep='\t', error_bad_lines=False)

        self.max_l = 0
        self.pad = pad
        self.transform = transform

        if whole_data is not None:
            vocab = self.get_vocab(whole_data)
            self.word_idx, self.pretrained_embs = self.load_word2vec(word_embedding_file, vocab, word_embedding_file.endswith('.bin'))
        # get embeddings size:
            self.k = len(self.pretrained_embs[0])
            print ("word2vec loaded (%d, %d)" % (len(self.word_idx), self.k))
            self.add_unknown_words(vocab)
            self.pretrained_embs = np.array(self.pretrained_embs)
        else:
            self.word_idx = word_idx
            self.pretrained_embs = pretrained_embs

    def load_word2vec(self,fname, vocab, binary=True):
        """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
        word_idx = {}
        pretrained_embs = []
        pretrained_embs.append(np.zeros((300,), dtype='float32'))
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
                    word_idx[word] = line + 1
                    brk()
                    pretrained_embs.append(np.fromstring(f.read(binary_len), dtype='float32'))
                    #if word in vocab:
                    #word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
                    #else:
                    #    f.read(binary_len)
            else: # text
                counter = 1
                for line in f:
                    items = line.split()
                    word = str(items[0], 'utf-8') #unicode(items[0], 'utf-8')
                    #word_vecs[word] = np.array(map(float, items[1:]))
                    word_idx[word] = counter
                    counter = counter + 1 
                    #brk()
                    pretrained_embs.append(np.array(items[1:], dtype='float')) #.append(np.array(map(float, items[1:])))
        return word_idx, pretrained_embs
    
    def get_vocab(self, whole_data, clean_string=False):
        vocab = defaultdict(int)
        whole_csv = pd.read_csv(whole_data, sep='\t', error_bad_lines=False)
        for (idx, row) in whole_csv.iterrows():
            if clean_string:
                clean_text = clean_str(row[3])
            else:
                clean_text = row[3].lower()
            words = clean_text.split()
            if self.max_l < len(words):
                self.max_l = len(words)
            for word in set(words):
                vocab[word] += 1           
        return vocab

    def add_unknown_words(self, vocab, min_df=1):
        """
        For words that occur in at least min_df documents, create a separate word vector.    
        0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
        :param k: size of embedding vectors.
        """
        counter = len(self.pretrained_embs)
        for word in vocab:
            if word not in self.word_idx and vocab[word] >= min_df:
                self.word_idx[word] = counter
                counter += 1
                self.pretrained_embs.append(np.random.uniform(-0.25, 0.25, self.k))


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        label = self.csv.iloc[idx, 2]
        sent = self.csv.iloc[idx, 3]
        
        x = [0 for i in range(self.pad)] 
        words = sent.split()[:self.max_l] # truncate words from test set
        for word in words:
            if word in self.word_idx: # FIXME: skips unknown words
                x.append(self.word_idx[word])
        while len(x) < self.max_l + 2 * self.pad : # right padding
            x.append(0)

        return  np.array(x), label, sent