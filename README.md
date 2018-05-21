# Sarcasm-Detection-using-CNN

This is the PyTorch implementation of work presented in 'Modelling Context with User Embeddings
for Sarcasm Detection in Social Media' (https://arxiv.org/pdf/1607.00976.pdf). The neural network takes in the input a tweet( content)and a unqiue user embedding(context), and classifies the tweets as sarcastic/non-sarcastic.

# System requirments
- python 2.7
- PyTorch 0.3.1


# Running the code
### Pre-requisites
Get pre-trained word embeddings (e.g. Skip-gram)
 1. Install the bin file from - 
 2. Run the iPython notebook - 
 3.


### Training and Evaluation
python train_CUE_CNN.py


# Output 
The code generate a progress folder, that contains sub folder for every run. Inside the run folder - 
1. logs.txt that contains loss and accuracy on train/test/validation set after every epoch
2. stats.jpg that plots -
  i. train/test/validation loss on a single plot
  ii. train/test/validation accuracy on a single plot





