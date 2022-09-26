# Sarcasm-Detection-using-HNN

This is the PyTorch implementation of work presented in 'Modelling Context with User Embeddings
for Sarcasm Detection in Social Media' (https://arxiv.org/pdf/1607.00976.pdf). The neural network takes a tweet (content) and corresponding user embedding (context) as input, and classifies the tweets as sarcastic/non-sarcastic. We further provide an implementation of our improved framework proposed in Sarcasm Detection using Hybrid Neural Network. (https://arxiv.org/abs/1908.07414).

## System requirments
- python 2.7
- PyTorch 0.3.1
- python package gensim
- python package yandex.translate
- python package ipdb


## Running the code
### 1. Pre-requisites
1. Get pre-trained word embeddings (e.g. Skip-gram)
   - Install the bin file from this [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
   - Unzip the .bin.gz fine and run the iPython notebook ```get_word2vec_embeddings.ipynb```
   - Place the .txt file obtained in ```DATA/embeddings/``` and change its name to ```words.txt```

2.  Get pre-trained user embeddings for the user. The embeddings we used can be found [here](https://www.dropbox.com/s/pmp5x08v6w09jrq/usr2vec_400_master.txt?dl=0). Place the embeddings in ```DATA/embeddings``` and name the file as ```usr2vec.txt```

3. Execute iPython notebook ```get_data.ipynb```. This utility code is used to download tweets corresponding to the tweet ids and then preprocess these tweet messages.

### 2. Training and Evaluation
#### a. To run the original code
Run ```python train_CUE_CNN.py```
#### b. To run the RNN + CNN Hybrid model on the new Dataset
Run ```python Headlines_RNN.py```

## Output, results and visualization 
The code generate a ```progress``` folder, that contains sub folder for every run. Inside every run folder following two file are generated - 
1. ```logs.txt``` which contains loss and accuracy on train/test/validation set after every epoch
2. ```stats.jpg``` that plots
   - train/test/validation loss on a single plot
   - train/test/validation accuracy on a single plot
   
## Note:
Util files, pre-trained user embeddings and raw tweet ids were obtained from [Original CUE-CNN](https://github.com/samiroid/CUE-CNN)

## Cite
Please cite the following articles in suitable format if you use the dataset:

**Text Format**:
```
1. Misra, Rishabh and Prahal Arora. "Sarcasm Detection using Hybrid Neural Network." arXiv preprint arXiv:1908.07414 (2019).
2. Misra, Rishabh and Jigyasa Grover. "Sculpting Data for ML: The first act of Machine Learning." ISBN 978-0-578-83125-1 (2021).
```
**BibTex Format**:
```
@article{misra2019sarcasm,
  title={Sarcasm Detection using Hybrid Neural Network},
  author={Misra, Rishabh and Arora, Prahal},
  journal={arXiv preprint arXiv:1908.07414},
  year={2019}
}

@book{misra2021sculpting,
author = {Misra, Rishabh and Grover, Jigyasa},
year = {2021},
month = {01},
pages = {},
title = {Sculpting Data for ML: The first act of Machine Learning},
isbn = {978-0-578-83125-1}
}
```

