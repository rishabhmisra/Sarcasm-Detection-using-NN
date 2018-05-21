"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""

# from my_utils.FMeasure import FmesSemEval
import numpy as np
import sys
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
#from theano.tensor.signal import downsample
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
from collections import OrderedDict
import time
import cPickle as pickle
from pdb import set_trace
from MLP import MLPDropout, ReLU,Sigmoid,Iden,HiddenLayer


# ----------------------------------------------------------------------
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        if self.non_linear == "none" or self.non_linear == "relu":
            self.W = theano.shared(np.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape), 
                                                dtype=theano.config.floatX),borrow=True,name="W_conv")
            # self.W = T.as_tensor_variable(np.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape), 
            #                                     dtype=theano.config.floatX),name="W_conv")
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype = theano.config.floatX),borrow=True,name="W_conv")   
            # self.W = T.as_tensor_variable(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            #     dtype = theano.config.floatX),name="W_conv")   
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        # self.b = T.as_tensor_variable(b_values, name="b_conv")
        
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear == "tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        elif self.non_linear == "relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = [self.W, self.b]
        
    def predict(self, new_data, batch_size):
        """
        predict for new data
        """
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear == "tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        if self.non_linear == "relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output
        

# ---------------------------------------------------------------------
class ConvNet(MLPDropout):
    """
    Adds convolution layers in front of a MLPDropout.
    """


    def __init__(self, E, U, height, width, filter_hs, conv_non_linear,
                 hidden_units, batch_size, non_static, dropout_rates,subspace_size=None,
                 activations=[Iden]):
        """
        height = sentence length (padded where necessary)
        width = word vector length (300 for word2vec)
        filter_hs = filter window sizes    
        hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
        """
        rng = np.random.RandomState(3435)
        feature_maps = hidden_units[0]
        self.batch_size = batch_size

        # define model architecture
        self.index = T.lscalar()
        self.x = T.matrix('x')   
        self.y = T.ivector('y')        
        self.Words = theano.shared(value=E, name="Words")   
        self.Users = None     
        self.u     = None
        self.subspace_size = subspace_size
        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(width)
        # reset Words to 0?
        self.set_zero = theano.function([zero_vec_tensor],
                                        updates=[(self.Words, T.set_subtensor(self.Words[0,:],zero_vec_tensor))],
                                        allow_input_downcast=True)
        # inputs to the ConvNet go to all convolutional filters:
        layer0_input = self.Words[T.cast(self.x.flatten(), dtype="int32")].reshape(
            (self.x.shape[0], 1, self.x.shape[1], self.Words.shape[1]))
        self.conv_layers = []       
        
        # outputs of convolutional filters
        layer1_inputs = []
        image_shape = (batch_size, 1, height, width)
        filter_w = width    
        for filter_h in filter_hs:            
            filter_shape = (feature_maps, 1, filter_h, filter_w)
            pool_size = (height-filter_h+1, width-filter_w+1)
            conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                            image_shape=image_shape,
                                            filter_shape=filter_shape,
                                            poolsize=pool_size,
                                            non_linear=conv_non_linear)
            layer1_input = conv_layer.output.flatten(2)
            self.conv_layers.append(conv_layer)
            layer1_inputs.append(layer1_input)
        # inputs to the MLP
        layer1_input = T.concatenate(layer1_inputs, 1)
        if U is not None:
            print "Will use user embeddings"
            self.u = T.ivector('u')
            self.Users = theano.shared(value=U, name="Users")
            them_users = self.Users[self.u]
            if self.subspace_size:
                print "and subspace"
                # set_trace()
                self.subspace = HiddenLayer(rng, them_users, U.shape[1], subspace_size, Sigmoid)
                self.peep = theano.function([self.x, self.u],[self.subspace.output,layer1_input],allow_input_downcast=True)

                layer1_input = T.concatenate((layer1_input,T.nnet.sigmoid(self.subspace.output)),1)
                layer_sizes = [feature_maps*len(filter_hs)+subspace_size]  
                # layer1_input = T.concatenate((layer1_input,them_users),1)
                # layer_sizes = [feature_maps*len(filter_hs)+U.shape[1]]

            else:
                layer1_input = T.concatenate((layer1_input,them_users),1)
                layer_sizes = [feature_maps*len(filter_hs)+U.shape[1]]

        else:
            print "NO user embeddings"
            layer_sizes = [feature_maps*len(filter_hs)]
        layer_sizes += hidden_units[1:]
        super(ConvNet, self).__init__(rng, input=layer1_input,
                                      layer_sizes=layer_sizes,
                                      activations=activations,
                                      dropout_rates=dropout_rates)

        # add parameters from convolutional layers
        for conv_layer in self.conv_layers:
            self.params += conv_layer.params
        if non_static:
            # if word vectors are allowed to change, add them as model parameters
            self.params += [self.Words]
        if U is not None:
            # if self.subspace_size is None:
                self.params += [self.Users]

    def predict(self, test_set_x):
        test_size = test_set_x.shape[0]
        height = test_set_x.shape[1]
        layer0_input = self.Words[T.cast(self.x.flatten(), dtype="int32")].reshape(
            (test_size, 1, height, self.Words.shape[1]))
        layer0_outputs = []
        for conv_layer in self.conv_layers:
            layer0_output = conv_layer.predict(layer0_input, test_size)
            layer0_outputs.append(layer0_output.flatten(2))
        layer1_input = T.concatenate(layer0_outputs, 1)

        if self.Users is not None:
            them_users = self.Users[self.u]
            # if self.subspace_size is not None:
            #     sub = T.nnet.sigmoid(T.dot(them_users,self.subspace.W))                 
            #     layer1_input = T.concatenate((layer1_input,sub),1)
            # else:
            #     layer1_input = T.concatenate((layer1_input,them_users),1)
            layer1_input = T.concatenate((layer1_input,them_users),1)

        return super(ConvNet, self).predict(layer1_input)

    def predict_prob(self, test_set_x):
        test_size = test_set_x.shape[0]
        height = test_set_x.shape[1]
        layer0_input = self.Words[T.cast(self.x.flatten(), dtype="int32")].reshape(
            (test_size, 1, height, self.Words.shape[1]))
        layer0_outputs = []
        for conv_layer in self.conv_layers:
            layer0_output = conv_layer.predict(layer0_input, test_size)
            layer0_outputs.append(layer0_output.flatten(2))
        layer1_input = T.concatenate(layer0_outputs, 1)

        if self.Users is not None:
            them_users = self.Users[self.u]
            # if self.subspace_size is not None:
            #     sub = T.nnet.sigmoid(T.dot(them_users,self.subspace.W))                 
            #     layer1_input = T.concatenate((layer1_input,sub),1)
            # else:
            #     layer1_input = T.concatenate((layer1_input,them_users),1)
            layer1_input = T.concatenate((layer1_input,them_users),1)

        return super(ConvNet, self).predict_p(layer1_input)

    # def train(self, train_set, shuffle_batch=True,
    #           epochs=25, lr_decay=0.95, sqr_norm_lim=9,labels=None,model=None):
    #     """
    #     Train a simple conv net
    #     sqr_norm_lim = s^2 in the paper
    #     lr_decay = adadelta decay parameter
    #     """    
    #     cost = self.negative_log_likelihood(self.y) 
    #     dropout_cost = self.dropout_negative_log_likelihood(self.y)
    #     # adadelta upgrades: dict of variable:delta
    #     grad_updates = self.sgd_updates_adadelta(dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    #     # shuffle dataset and assign to mini batches.
    #     # if dataset size is not a multiple of batch size, replicate 
    #     # extra data (at random)
    #     np.random.seed(3435)
    #     batch_size = self.batch_size
    #     if train_set.shape[0] % batch_size > 0:
    #         extra_data_num = batch_size - train_set.shape[0] % batch_size
    #         #extra_data = train_set[np.random.choice(train_set.shape[0], extra_data_num)]
    #         perm_set = np.random.permutation(train_set)   
    #         extra_data = perm_set[:extra_data_num]
    #         new_data = np.append(train_set, extra_data, axis=0)
    #     else:
    #         new_data = train_set
        
    #     shuffled_data = np.random.permutation(new_data) # Attardi
    #     n_batches     = shuffled_data.shape[0]/batch_size
    #     # divide train set into 90% train, 10% validation sets
    #     n_train_batches = int(np.round(n_batches*0.8))
    #     n_val_batches = n_batches - n_train_batches
    #     train_set = shuffled_data[:n_train_batches*batch_size,:]
    #     val_set   = shuffled_data[n_train_batches*batch_size:,:]     
    #     # push data to gpu        
    #     # the dataset has the format [word_indices,padding,user,label]
    #     train_set_x, train_set_y = shared_dataset(train_set[:,:-2], train_set[:,-1])  
    #     train_set_u = theano.shared(np.asarray(train_set[:,-2],dtype='int32'))      
    #     # val_set_x = val_set[:,:-2]
    #     # val_set_u = val_set[:,-2]
    #     # val_set_y = val_set[:,-1]

    #     val_set_x, val_set_y = shared_dataset(val_set[:,:-2], val_set[:,-1])
    #     val_set_u  = theano.shared(np.asarray(val_set[:,-2],dtype='int32'))      

    #     # val_set_x, val_set_y = shared_dataset(val_set[:,:-1], val_set[:,-1])
    #     batch_start = self.index * batch_size
    #     batch_end = batch_start + batch_size

        
        
    #     # compile Theano functions to get train/val/test errors
        
    #     # val_model = theano.function([self.index], self.errors(self.y),
    #     #                             givens={
    #     #                                 self.x: val_set_x[batch_start:batch_end],
    #     #                                 self.y: val_set_y[batch_start:batch_end]},
    #     #                             allow_input_downcast=True)

    #     #FIXME: this is a bit weird
    #     test_y_pred = self.predict(val_set_x,val_set_u)
    #     # make_preds  = theano.function([self.x], test_y_pred, allow_input_downcast=True)

    #     test_error = T.mean(T.neq(test_y_pred, self.y))
        

    #     # errors on train set
    #     if self.Users is not None:
    #         train_model = theano.function([self.index], cost, updates=grad_updates,
    #                                   givens={
    #                                       self.x: train_set_x[batch_start:batch_end],
    #                                       self.y: train_set_y[batch_start:batch_end],
    #                                       self.u: train_set_u[batch_start:batch_end]
    #                                       },
    #                                   allow_input_downcast = True)

    #         train_error = theano.function([self.index], self.errors(self.y),
    #                                       givens={
    #                                           self.x: train_set_x[batch_start:batch_end],
    #                                           self.y: train_set_y[batch_start:batch_end],
    #                                           self.u: train_set_u[batch_start:batch_end]},
    #                                       allow_input_downcast=True)
    #         val_model = theano.function([self.index], self.errors(self.y),
    #                                 givens={
    #                                     self.x: val_set_x[batch_start:batch_end],
    #                                     self.y: val_set_y[batch_start:batch_end],        
    #                                     self.u: val_set_u[batch_start:batch_end]},
    #                                 allow_input_downcast=True)
    #         test_model = theano.function([self.x, self.y,self.u], test_error, allow_input_downcast=True)
    #     else:
    #         train_model = theano.function([self.index], cost, updates=grad_updates,
    #                                   givens={
    #                                       self.x: train_set_x[batch_start:batch_end],
    #                                       self.y: train_set_y[batch_start:batch_end]},
    #                                   allow_input_downcast = True)

    #         train_error = theano.function([self.index], self.errors(self.y),
    #                                       givens={
    #                                           self.x: train_set_x[batch_start:batch_end],
    #                                           self.y: train_set_y[batch_start:batch_end]},
    #                                       allow_input_downcast=True)
    #         val_model = theano.function([self.index], self.errors(self.y),
    #                                 givens={
    #                                     self.x: val_set_x[batch_start:batch_end],
    #                                     self.y: val_set_y[batch_start:batch_end],        
    #                                    },
    #                                 allow_input_downcast=True)
    #         test_model = theano.function([self.x, self.y], test_error, allow_input_downcast=True)

        


    #     # start training over mini-batches
    #     print 'training...'        
    #     best_val_perf = 0
    #     test_perf = 0    
    #     patience = 5
    #     drops    = 0
    #     prev_val_perf = 0  
    #     for epoch in xrange(epochs):
    #         start_time = time.time()
    #         # FIXME: should permute whole set rather than minibatch indexes
    #         if shuffle_batch:
    #             for minibatch_index in np.random.permutation(range(n_train_batches)):
    #                 cost_epoch = train_model(minibatch_index)
    #                 self.set_zero(self.zero_vec) # CHECKME: Why?
    #         else:
    #             for minibatch_index in xrange(n_train_batches):
    #                 cost_epoch = train_model(minibatch_index)  
    #                 self.set_zero(self.zero_vec)
    #         train_losses = [train_error(i) for i in xrange(n_train_batches)]
    #         train_perf = 1 - np.mean(train_losses)
    #         val_losses = [val_model(i) for i in xrange(n_val_batches)]
    #         val_perf = 1 - np.mean(val_losses)     
    #         # test_loss = test_model(val_set_x, val_set_y)
    #         # test_perf = 1 - test_loss         
    #         # predz = make_preds(val_set_x)
    #         # val_perf = FmesSemEval(predz, val_set_y, pos_ind, neg_ind)
    #         # val_perf = 0
    #         info = 'epoch: %i\%i (%.2f secs) train acc: %.2f %% | val acc: %.2f %%' % (
    #             epoch,epochs, time.time()-start_time, train_perf * 100., val_perf*100.)              
    #         # from ipdb import set_trace; set_trace()
    #         if val_perf > prev_val_perf:                
    #             drops=0
    #             if val_perf >= best_val_perf:
    #                 best_val_perf = val_perf
    #                 info+= " **"
    #                 if model:
    #                     # print "save model"
    #                     self.save(model)
    #                 # test_loss = test_wmodel(val_set_x, val_set_y)
    #                 # test_perf = 1 - test_loss         
    #                 # predz = make_preds(val_set_x)
    #                 # fmes = FmesSemEval(predz, val_set_y, pos_ind, neg_ind)
    #                 # print predz
    #                 # print test_set_y
    #                 # print "Test performance acc: %.3f | polar fmes:%.3f " % (test_perf,fmes)
    #         else: 
    #             drops+=1
    #         if drops >= patience:
    #             print "Ran out of patience..."
    #             break
    #         prev_val_perf = val_perf
    #         print info
    #     return test_perf

    def evaluate(self, train_set, test_set, shuffle_batch=True,
              epochs=25, lr_decay=0.95, sqr_norm_lim=9,labels=None,model=None):
        """
        Train a simple conv net
        sqr_norm_lim = s^2 in the paper
        lr_decay = adadelta decay parameter
        """    
        cost = self.negative_log_likelihood(self.y) 
        dropout_cost = self.dropout_negative_log_likelihood(self.y)
        # adadelta upgrades: dict of variable:delta
        grad_updates = self.sgd_updates_adadelta(dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
        # shuffle dataset and assign to mini batches.
        # if dataset size is not a multiple of batch size, replicate 
        # extra data (at random)
        np.random.seed(3435)
        batch_size = self.batch_size
        if train_set.shape[0] % batch_size > 0:
            extra_data_num = batch_size - train_set.shape[0] % batch_size
            #extra_data = train_set[np.random.choice(train_set.shape[0], extra_data_num)]
            perm_set = np.random.permutation(train_set)   
            extra_data = perm_set[:extra_data_num]
            new_data = np.append(train_set, extra_data, axis=0)
        else:
            new_data = train_set
        
        #shuffled_data = np.random.permutation(new_data) # Attardi
        shuffled_data = new_data
        n_batches     = shuffled_data.shape[0]/batch_size
        # divide train set into 90% train, 10% validation sets
        n_train_batches = int(np.round(n_batches*0.9))
        n_val_batches = n_batches - n_train_batches
        train_set = shuffled_data[:n_train_batches*batch_size,:]
        val_set   = shuffled_data[n_train_batches*batch_size:,:]     
        # push data to gpu        
        # the dataset has the format [word_indices,padding,user,label]
        train_set_x, train_set_y = shared_dataset(train_set[:,:-2], train_set[:,-1])  
        train_set_u = theano.shared(np.asarray(train_set[:,-2],dtype='int32'))      
        # val_set_x = val_set[:,:-2]
        # val_set_u = val_set[:,-2]
        # val_set_y = val_set[:,-1]
        val_set_x, val_set_y = shared_dataset(val_set[:,:-2], val_set[:,-1])
        val_set_u  = theano.shared(np.asarray(val_set[:,-2],dtype='int32'))      
        test_set_x = test_set[:,:-2]
        test_set_u = test_set[:,-2]
        test_set_y = test_set[:,-1]        
        batch_start = self.index * batch_size
        batch_end = batch_start + batch_size

        # compile Theano functions to get train/val/test errors
        
        
        test_y_pred = self.predict(test_set_x)
        test_error = T.mean(T.neq(test_y_pred, self.y))
        # errors on train set
        if self.Users is not None:
            train_model = theano.function([self.index], cost, updates=grad_updates,
                                      givens={
                                          self.x: train_set_x[batch_start:batch_end],
                                          self.y: train_set_y[batch_start:batch_end],
                                          self.u: train_set_u[batch_start:batch_end]
                                          },
                                      allow_input_downcast = True)

            train_error = theano.function([self.index], self.errors(self.y),
                                          givens={
                                              self.x: train_set_x[batch_start:batch_end],
                                              self.y: train_set_y[batch_start:batch_end],
                                              self.u: train_set_u[batch_start:batch_end]},
                                          allow_input_downcast=True)
            val_model = theano.function([self.index], self.errors(self.y),
                                    givens={
                                        self.x: val_set_x[batch_start:batch_end],
                                        self.y: val_set_y[batch_start:batch_end],        
                                        self.u: val_set_u[batch_start:batch_end]},
                                    allow_input_downcast=True)
            test_model = theano.function([self.x, self.u, self.y], test_error, allow_input_downcast=True)
        else:
            train_model = theano.function([self.index], cost, updates=grad_updates,
                                      givens={
                                          self.x: train_set_x[batch_start:batch_end],
                                          self.y: train_set_y[batch_start:batch_end]},
                                      allow_input_downcast = True)

            train_error = theano.function([self.index], self.errors(self.y),
                                          givens={
                                              self.x: train_set_x[batch_start:batch_end],
                                              self.y: train_set_y[batch_start:batch_end]},
                                          allow_input_downcast=True)

            val_model = theano.function([self.index], self.errors(self.y),
                                    givens={
                                        self.x: val_set_x[batch_start:batch_end],
                                        self.y: val_set_y[batch_start:batch_end]},
                                    allow_input_downcast=True)
            test_model = theano.function([self.x, self.y], test_error, allow_input_downcast=True)

        # start training over mini-batches
        print 'training...' 
        best_val_perf = 0
        test_perf = 0    
        patience = 5
        drops    = 0
        prev_val_perf = 0  
        for epoch in xrange(epochs):
#             if epoch == 4:
#                 break
            start_time = time.time()
            # FIXME: should permute whole set rather than minibatch indexes
            if shuffle_batch:
                count = 0
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    cost_epoch = train_model(minibatch_index)
                    self.set_zero(self.zero_vec) # CHECKME: Why?
                    count = count + 1
                    #print count, time.time() - start_time
                    #sys.stdout.flush()
                    #if count > 10:
                    #    print "...", n_train_batches, shuffle_batch, n_batches, batch_size
                    #    sys.exit()
            else:
                for minibatch_index in xrange(n_train_batches):
                    cost_epoch = train_model(minibatch_index)  
                    self.set_zero(self.zero_vec)
                    #print "...", n_train_batches, shuffle_batch, n_batches, batch_size
                    #sys.exit()
            train_losses = [train_error(i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [val_model(i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.mean(val_losses)     
            info = 'epoch: %i\%i (%.2f secs) train acc: %.2f %% | val acc: %.2f %%' % (
                epoch,epochs, time.time()-start_time, train_perf * 100., val_perf*100.)              
            # from ipdb import set_trace; set_trace()
            if val_perf > prev_val_perf:                
                drops=0
                if val_perf >= best_val_perf:
                    best_val_perf = val_perf
                    info+= " **"
                    if model:
                        # print "save model"
                        self.save(model)
                    if self.Users is not None:
                        test_loss = test_model(test_set_x, test_set_u, test_set_y)
                    else:
                        test_loss = test_model(test_set_x, test_set_y)
                    test_perf = 1 - test_loss         
            else: 
                drops+=1
            if drops >= patience:
                print "Ran out of patience..."
                break
            prev_val_perf = val_perf
            print info
            print "Test acc: %.2f %% " % (test_perf * 100.)
            sys.stdout.flush()
        # set_trace() 
        return test_perf

    def train(self, train_set, shuffle_batch=True,
              epochs=25, lr_decay=0.95, sqr_norm_lim=9,labels=None,model=None):
        """
        Train a simple conv net
        sqr_norm_lim = s^2 in the paper
        lr_decay = adadelta decay parameter
        """    
        cost = self.negative_log_likelihood(self.y) 
        dropout_cost = self.dropout_negative_log_likelihood(self.y)
        # adadelta upgrades: dict of variable:delta
        grad_updates = self.sgd_updates_adadelta(dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
        # shuffle dataset and assign to mini batches.
        # if dataset size is not a multiple of batch size, replicate 
        # extra data (at random)
        np.random.seed(3435)
        batch_size = self.batch_size
        if train_set.shape[0] % batch_size > 0:
            extra_data_num = batch_size - train_set.shape[0] % batch_size
            #extra_data = train_set[np.random.choice(train_set.shape[0], extra_data_num)]
            perm_set = np.random.permutation(train_set)   
            extra_data = perm_set[:extra_data_num]
            new_data = np.append(train_set, extra_data, axis=0)
        else:
            new_data = train_set
        
        shuffled_data = np.random.permutation(new_data) # Attardi
        n_batches     = shuffled_data.shape[0]/batch_size
        # divide train set into 90% train, 10% validation sets
        n_train_batches = int(np.round(n_batches*0.8))
        n_val_batches = n_batches - n_train_batches
        train_set = shuffled_data[:n_train_batches*batch_size,:]
        val_set   = shuffled_data[n_train_batches*batch_size:,:]     
        # push data to gpu        
        # the dataset has the format [word_indices,padding,user,label]
        train_set_x, train_set_y = shared_dataset(train_set[:,:-2], train_set[:,-1])  
        train_set_u = theano.shared(np.asarray(train_set[:,-2],dtype='int32'))      
        # val_set_x = val_set[:,:-2]
        # val_set_u = val_set[:,-2]
        # val_set_y = val_set[:,-1]
        val_set_x, val_set_y = shared_dataset(val_set[:,:-2], val_set[:,-1])
        val_set_u  = theano.shared(np.asarray(val_set[:,-2],dtype='int32'))      
        
        batch_start = self.index * batch_size
        batch_end = batch_start + batch_size

        # compile Theano functions to get train/val/test errors
        
        
        
        # errors on train set
        if self.Users is not None:
            train_model = theano.function([self.index], cost, updates=grad_updates,
                                      givens={
                                          self.x: train_set_x[batch_start:batch_end],
                                          self.y: train_set_y[batch_start:batch_end],
                                          self.u: train_set_u[batch_start:batch_end]
                                          },
                                      allow_input_downcast = True)

            train_error = theano.function([self.index], self.errors(self.y),
                                          givens={
                                              self.x: train_set_x[batch_start:batch_end],
                                              self.y: train_set_y[batch_start:batch_end],
                                              self.u: train_set_u[batch_start:batch_end]},
                                          allow_input_downcast=True)
            val_model = theano.function([self.index], self.errors(self.y),
                                    givens={
                                        self.x: val_set_x[batch_start:batch_end],
                                        self.y: val_set_y[batch_start:batch_end],        
                                        self.u: val_set_u[batch_start:batch_end]},
                                    allow_input_downcast=True)
            
        else:
            train_model = theano.function([self.index], cost, updates=grad_updates,
                                      givens={
                                          self.x: train_set_x[batch_start:batch_end],
                                          self.y: train_set_y[batch_start:batch_end]},
                                      allow_input_downcast = True)

            train_error = theano.function([self.index], self.errors(self.y),
                                          givens={
                                              self.x: train_set_x[batch_start:batch_end],
                                              self.y: train_set_y[batch_start:batch_end]},
                                          allow_input_downcast=True)

            val_model = theano.function([self.index], self.errors(self.y),
                                    givens={
                                        self.x: val_set_x[batch_start:batch_end],
                                        self.y: val_set_y[batch_start:batch_end]},
                                    allow_input_downcast=True)

        # start training over mini-batches
        print 'training...'        
        best_val_perf = 0        
        patience = 5
        drops    = 0
        prev_val_perf = 0  
        for epoch in xrange(epochs):
            start_time = time.time()
            # FIXME: should permute whole set rather than minibatch indexes
            if shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    cost_epoch = train_model(minibatch_index)
                    self.set_zero(self.zero_vec) # CHECKME: Why?
            else:
                for minibatch_index in xrange(n_train_batches):
                    cost_epoch = train_model(minibatch_index)  
                    self.set_zero(self.zero_vec)
            train_losses = [train_error(i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [val_model(i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.mean(val_losses)     
            info = 'epoch: %i\%i (%.2f secs) train acc: %.2f %% | val acc: %.2f %%' % (
                epoch,epochs, time.time()-start_time, train_perf * 100., val_perf*100.)              
            # from ipdb import set_trace; set_trace()
            if val_perf > prev_val_perf:                
                drops=0
                if val_perf >= best_val_perf:
                    best_val_perf = val_perf
                    info+= " **"
                    if model:
                        # print "save model"
                        self.save(model)                    
            else: 
                drops+=1
            if drops >= patience:
                print "Ran out of patience..."
                break
            prev_val_perf = val_perf
            print info
        # set_trace() 
        return best_val_perf

    def sgd_updates_adadelta(self, cost, rho=0.95, epsilon=1e-6, norm_lim=9,
                             word_vec_name='Words'):
        """
        adadelta update rule, mostly from
        https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
        :retuns: a dictionary of variable:delta
        """
        updates = OrderedDict({})
        exp_sqr_grads = OrderedDict({})
        exp_sqr_ups = OrderedDict({})
        gparams = []
        for param in self.params:
            empty = np.zeros_like(param.get_value())
            exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),
                                                 name="exp_grad_%s" % param.name)
            gp = T.grad(cost, param)
            exp_sqr_ups[param] = theano.shared(value=as_floatX(empty),
                                               name="exp_grad_%s" % param.name)
            gparams.append(gp)
        for param, gp in zip(self.params, gparams):
            exp_sg = exp_sqr_grads[param]
            exp_su = exp_sqr_ups[param]
            up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
            updates[exp_sg] = up_exp_sg
            step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
            updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
            stepped_param = param + step
            if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
                col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
                desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
                scale = desired_norms / (1e-7 + col_norms)
                updates[param] = stepped_param * scale
            else:
                updates[param] = stepped_param      
        return updates 


    
    # def predict(self, test_set_x):
    #     test_size = test_set_x.shape[0]
    #     height = test_set_x.shape[1]
    #     layer0_input = self.Words[T.cast(self.x.flatten(), dtype="int32")].reshape(
    #         (test_size, 1, height, self.Words.shape[1]))
    #     layer0_outputs = []
    #     for conv_layer in self.conv_layers:
    #         layer0_output = conv_layer.predict(layer0_input, test_size)
    #         layer0_outputs.append(layer0_output.flatten(2))
    #     layer1_input = T.concatenate(layer0_outputs, 1)
        # return super(ConvNet, self).predict(layer1_input)


    def save(self, mfile):
        """
        Save network params to file.
        """
        # set_trace()
        pickle.dump((self.params, self.layers, self.conv_layers),
                    mfile, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load(cls, mfile,users=False):
        cnn = cls.__new__(cls)
        cnn.params, cnn.layers, cnn.conv_layers = pickle.load(mfile)
        if users:
            cnn.Words = cnn.params[-2]
            cnn.Users = cnn.params[-1]
            cnn.u = T.ivector('u')
        else:
            cnn.Words = cnn.params[-1]
        cnn.index = T.lscalar()
        cnn.x = T.matrix('x')   
        cnn.y = T.ivector('y')
        return cnn

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=np.int32),
                                 borrow=borrow)
        return shared_x, shared_y

