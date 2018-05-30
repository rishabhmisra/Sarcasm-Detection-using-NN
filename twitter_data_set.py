import ast
import csv
import sys
import numpy as np
from torch.utils.data.dataset import Dataset
from collections import defaultdict
from pdb import set_trace as brk
import torch

class TwitterDataset(Dataset):
    """Twitter dataset."""

    def __init__(self, csv_file, folds_file, word_embedding_file, user_embedding_file, set_type, pad, w2v=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with tweets.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        csv.field_size_limit(sys.maxsize)
        with open(folds_file) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.data_ids = ast.literal_eval(row[set_type])
        
        #self.tweets = pd.read_csv(csv_file)
        
        print "loading data..."
        self.max_l = 0
        self.pad = pad
        self.sents, vocab, users = self.build_data(csv_file, self.data_ids)
        print "data loaded!"
        print "number of sentences: " + str(len(self.sents))
        print "vocab size: " + str(len(vocab))
        print "max sentence length: " + str(self.max_l)
        
        print "loading word2vec vectors..."
        
        if set_type == 'train':
            w2v = self.load_word2vec(word_embedding_file, vocab, word_embedding_file.endswith('.bin'))
            # get embeddings size:
            k = len(w2v.itervalues().next())
            print "word2vec loaded (%d, %d)" % (len(w2v), k)
            self.add_unknown_words(w2v, vocab, k)
        self.w2v = w2v

        u2v = self.load_word2vec(user_embedding_file, users, False)        
        j = len(u2v.itervalues().next())
        print "user2vec loaded (%d, %d)" % (len(u2v), j)
        self.add_unknown_words(u2v, users, j)
        self.u2v = u2v

        self.transform = transform

    def build_data(self, train_file, data_ids, clean_string=False, tagField=2, textField=3, userField=1, idField=0):
        """
        Loads data and split into 10 folds.
        :return: sents (with class and split properties), word doc freq, list of labels.
        """
        revs = {}
        vocab = defaultdict(int)
        users = {}
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
                user = fields[userField]               
                msg_id = fields[idField]

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
                         "y": int(tag),
                         "text": clean_text,
                         "num_words": len(words),
                         "msg_id" : msg_id
                         }
                if int(datum["msg_id"]) in data_ids:
                    revs[int(datum["msg_id"])] = datum
                self.max_l = max(self.max_l, len(words))

        return revs, vocab, users

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
        return len(self.data_ids)

    def __getitem__(self, idx):
        #landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        msg_id = self.data_ids[idx]
        label = self.sents[msg_id]['y']
        user_embedding = self.u2v[self.sents[msg_id]['user']]
        
        k = self.w2v.itervalues().next()
        x = [[0] * k for i in range(self.pad)] 
        words = self.sents[msg_id]['text'].split()[:self.max_l] # truncate words from test set
        for word in words:
            if word in self.w2v: # FIXME: skips unknown words
                x.append(self.w2v[word])
        while len(x) < self.max_l + 2 * self.pad : # right padding
            x.append([0] * k)

#         if self.transform:
#             data_point = self.transform(data_point)

        return  torch.from_numpy(np.array(x)).unsqueeze(0), user_embedding, label, self.sents[msg_id]['text']