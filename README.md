# Sarcasm-Detection-using-CNN

This is the PyTorch implementation of work presented in 'Modelling Context with User Embeddings
for Sarcasm Detection in Social Media' (https://arxiv.org/pdf/1607.00976.pdf). The neural network takes in the input a tweet( content)and a unqiue user embedding(context), and classifies the tweets as sarcastic/non-sarcastic.

## System requirments
- python 2.7
- PyTorch 0.3.1


## Running the code
### 1. Pre-requisites
1. Get pre-trained word embeddings (e.g. Skip-gram)
   - Install the bin file from this [link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
   - Unzip the .bin.gz fine and run the iPython notebook ```..........```
   - Place the .txt file obtained in ```DATA/embeddings/``` and change its name to ```words.txt```

2.  Get pre-trained user embeddings for the user. The embeddings we used can be found [here](https://www.dropbox.com/s/pmp5x08v6w09jrq/usr2vec_400_master.txt?dl=0). Place the embeddings in ```DATA/embeddings``` and name the file as ```usr2vec.txt```

3.  Get the user tweets - To comply with Twitter policies we can only share the msg ids. These can be found in the file bamman_redux_ids.txt

4. Clone or download the [my_utils](https://github.com/samiroid/utils) module and place it under the folder ```code```. Execute iPython notebook ```notebook.ipynb```. This utility code is used to download tweets from the ids and then preprocess these tweet messages.

### 2. Training and Evaluation
Run ```python train_CUE_CNN.py```

## Output, results and visualization 
The code generate a progress folder, that contains sub folder for every run. Inside the run folder - 
1. logs.txt that contains loss and accuracy on train/test/validation set after every epoch
2. stats.jpg that plots
   - train/test/validation loss on a single plot
   - train/test/validation accuracy on a single plot
