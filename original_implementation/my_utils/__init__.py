from collections import Counter, defaultdict
import csv
from ipdb import set_trace
import numpy as np
import os
import re
import sys
import twokenize
from tweetokenize import Tokenizer
from yandex_translate import YandexTranslate, YandexTranslateException

# emoticon regex taken from Christopher Potts' script at http://sentiment.christopherpotts.net/tokenizing.html
emoticon_regex = r"""(?:[<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]|[\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?)"""

twk = Tokenizer(ignorequotes=False,usernames=False,urls=False)

def count_emoticon_polarity(message):
    """
        returns the number of positive, neutral and negative emoticons in message
    """
    emoticon_list = re.findall(emoticon_regex, message)
    polarity_list = []
    for emoticon in emoticon_list:
        if emoticon in ['8:', '::', 'p:']:
            continue # these are false positives: '8:48', 'http:', etc
        polarity = emoticon_polarity(emoticon)
        polarity_list.append(polarity)          
    emoticons = Counter(polarity_list)
    pos = emoticons[1]
    neu = emoticons[0]
    neg = emoticons[-1]
    
    return pos,neu,neg

def remove_emoticons(message):
    return re.sub(emoticon_regex,'',message)

def emoticon_polarity(emoticon):
    
    eyes_symbol = re.findall(r'[:;=8]', emoticon) # find eyes position    
    #if valid eyes are not found return 0
    if len(eyes_symbol) == 1:
        eyes_symbol = eyes_symbol[0]    
    else:
        return 0
    mouth_symbol = re.findall(r'[\)\]\(\[dDcCpP/\}\{@\|\\]', emoticon) # find mouth position    
    #if a valid mouth is not found return 0
    if len(mouth_symbol) == 1:
        mouth_symbol = mouth_symbol[0]
    else:
        return 0
    eyes_index = emoticon.index(eyes_symbol)
    mouth_index = emoticon.index(mouth_symbol)
    # this assumes typical smileys like :)
    if mouth_symbol in [')', ']', '}', 'D', 'd']:
        polarity = +1
    elif mouth_symbol in ['(', '[', '{', 'C', 'c']:
        polarity = -1
    elif mouth_symbol in ['p', 'P', '\\', '/', ':', '@', '|']:
        polarity = 0
    else:
        raise Exception                
    # now we reverse polarity for reversed smileys like (:
    if eyes_index > mouth_index:
        polarity = -polarity

    return polarity
  
def colstr(string, color, best):
    # set_trace()
    if color is None:
        cstring = string
    elif color == 'red':
        cstring = "\033[31m" + string  + "\033[0m"
    elif color == 'green':    
        cstring = "\033[32m" + string  + "\033[0m"

    if best: 
        cstring += " ** "
    else:
        cstring += "    "

    return cstring    
    
def max_reps(sentence, n=3):

    """
        Normalizes a string to at most n repetitions of the same character
        e.g, for n=3 and "helllloooooo" -> "helllooo"
    """
    new_sentence = ''
    last_c = ''
    max_counter = n
    for c in sentence:
        if c != last_c:
            new_sentence+=c
            last_c = c
            max_counter = n
        else:
            if max_counter > 1:
                new_sentence+=c
                max_counter-=1
            else:
                pass
    return new_sentence

def word_2_idx(msgs, zero_for_padd=True, max_words=None):
    """
        Compute a dictionary index mapping words into indices
    """ 
    words = [w for m in msgs for w in m.split()]
    if max_words is not None:                
        top_words = sorted(Counter(words).items(), key=lambda x:x[1],reverse=True)[:max_words]                    
        words = [w[0] for w in top_words]
    #prepend the padding token
    if zero_for_padd: words = ['_pad_'] + list(words)    
    return {w:i for i,w in enumerate(set(words))}

def preprocess(m, sep_emoji=False):
    m = m.lower()    
    m = max_reps(m)
    #replace user mentions with token '@user'
    user_regex = r".?@.+?( |$)|<@mention>"    
    m = re.sub(user_regex," @user ", m, flags=re.I)
    #replace urls with token 'url'
    m = re.sub(twokenize.url," url ", m, flags=re.I)        
    tokenized_msg = ' '.join(twokenize.tokenize(m)).strip()
    if sep_emoji:
        #tokenize emoji, this tokenzier however has a problem where repeated punctuation gets separated e.g. "blah blah!!!"" -> ['blah','blah','!!!'], instead of ['blah','blah','!','!','!']
        m_toks = tokenized_msg.split()
        n_toks = twk.tokenize(tokenized_msg)         
        if len(n_toks)!=len(m_toks):
            #check if there is any punctuation in this string
            has_punct = map(lambda x:x in twk.punctuation, n_toks)
            if any(has_punct):  
                new_m = n_toks[0]
                for i in xrange(1,len(n_toks)):
                    #while the same punctuation token shows up, concatenate
                    if has_punct[i] and has_punct[i-1] and (n_toks[i] == n_toks[i-1]):
                        new_m += n_toks[i]
                    else:
                        #otherwise add space
                        new_m += " "+n_toks[i]                   
                tokenized_msg = new_m                
    return tokenized_msg.lstrip()

def preprocess_corpus(corpus_in, corpus_out, max_sent=float('inf'), sep_emoji=False):

    with open(corpus_out,"w") as fod:    
        with open(corpus_in) as fid:
            for i, l in enumerate(fid):
                if i > max_sent:
                    break
                elif not i%1000:
                    sys.stdout.write("\ri:%d" % i)
                    sys.stdout.flush()
                nl = preprocess(l.decode("utf-8"),sep_emoji)
                # set_trace()
                fod.write(nl.encode("utf-8")+"\n")
    print "\nprocessed corpus @ %s " % corpus_out

def kfolds(n_folds,n_elements,val_set=False,shuffle=False,random_seed=1234):        
    if val_set:
        assert n_folds>2
    
    X = np.arange(n_elements)
    if shuffle: 
        rng=np.random.RandomState(random_seed)      
        rng.shuffle(X)    
    X = X.tolist()
    slice_size = n_elements/n_folds
    slices =  [X[j*slice_size:(j+1)*slice_size] for j in xrange(n_folds)]
    #append the remaining elements to the last slice
    slices[-1] += X[n_folds*slice_size:]
    kf = []
    for i in xrange(len(slices)):
        train = slices[:]
        # from pdb import set_trace; set_trace()
        # print i
        test = train.pop(i)
        if val_set:
            try:
                val = train.pop(i)
            except IndexError:
                val = train.pop(-1)                
            #flatten the list of lists
            train = [item for sublist in train for item in sublist]
            kf.append([train,test,val])
        else:
            train = [item for sublist in train for item in sublist]
            kf.append([train,test])
    return kf

def build_folds(msg_ids, n_folds, folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    kf = kfolds(n_folds, len(msg_ids),val_set=True,shuffle=True)
    for i, fold in enumerate(kf):
        fold_data = {"train": [ int(msg_ids[x]) for x in fold[0] ], 
                     "test" : [ int(msg_ids[x]) for x in fold[1] ],
                     "val"  : [ int(msg_ids[x]) for x in fold[2] ] }
        with open(folder_path + '/fold_%d.csv' % i, 'wb') as f: 
            w = csv.DictWriter(f, fold_data.keys())
            w.writeheader()
            w.writerow(fold_data)

def shuffle_split(data, split_perc = 0.8, random_seed=1234):
    """
        Split the data into train and test, keeping the class proportions

        data: list of (x,y) tuples
        split_perc: percentage of training examples in train/test split
        random_seed: ensure repeatable shuffles

        returns: balanced training and test sets
    """
    rng=np.random.RandomState(random_seed)          
    z = defaultdict(list)
    #shuffle data
    rng.shuffle(data)
    #group examples by class label    
    z = defaultdict(list)
    for x,y in data: z[y].append(x)    
    train = []    
    test  = []
    for label in z.keys():
        #examples of each label 
        x_label  = z[label]            
        train += zip(x_label[:int(len(x_label)*split_perc)],
                    [label] * int(len(x_label)*split_perc))         
        test  += zip(x_label[ int(len(x_label)*split_perc):],
                    [label] * int(len(x_label)*(1-split_perc)))
    #reshuffle
    rng.shuffle(train)
    rng.shuffle(test)    

    return train, test

def translate_corpus(api_key, translation_pair,
                     path_in, path_out, resume_from=0,
                     max_sent=float('inf')):    
    """
        Translate a corpus using Yandex API
        api_key: API key
        translation_pair: FROM_LANGUAGE-TO_LANGUAGE, e.g.: "en-es"
        path_in: corpus to be translated
        path_out: path where the translation will be saved
        resume_from: number of the line on the input corpus from which to start translating;
                     this allows for translations to be resumed in case of failure; 
                     NOTE: if this is set the path_out will be open to append
    """
    translator = YandexTranslate(api_key)
    fails = []
    if resume_from > 0:
        mode = "a"        
    else:
        mode = "w"
    print "open in mode ", mode
    with open(path_out, mode) as fod:    
        with open(path_in) as fid:
            if resume_from > 0:
                print "skipping the first %d lines" % resume_from                
                #skip the first lines
                for _ in xrange(resume_from): next(fid)
            for i, l in enumerate(fid):
                if i > max_sent:
                    break
                elif not i%100:
                    sys.stdout.write("\r> %d" % i)
                    sys.stdout.flush() 
                #request translation
                try:
                    tr = translator.translate(l,translation_pair)
                    #grab the tex
                    txt = tr['text'][0].encode("utf-8")                                        
                except YandexTranslateException as e:                     
                    #check response codes
                    if e.message == 'ERR_KEY_INVALID':
                        print "\n[ABORTED: Invalid API key]\n"
                        break
                    elif e.message == 'ERR_KEY_BLOCKED':
                        print "\n[ABORTED: Blocked API key]\n"
                        break
                    elif e.message == 'ERR_DAILY_CHAR_LIMIT_EXCEEDED':
                        print "\n[ABORTED: Exceeded daily limit]\n"
                        break
                    elif e.message == 'ERR_LANG_NOT_SUPPORTED':
                        print "\n[ABORTED: Translation direction not supported]\n"
                        break
                    elif e.message == 'ERR_UNPROCESSABLE_TEXT' or tr['code'] == 'ERR_TEXT_TOO_LONG':
                        #413 - exceeded maximum text size
                        #422 - text cannot be translated
                        txt = "FAIL: ", e.message
                        fails.append(l)                                                      
                fod.write(txt)
    print "\ntranslated corpus @ %s " % path_out
    print "fails", fails
    