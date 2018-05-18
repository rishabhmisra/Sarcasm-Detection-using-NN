import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time

from pdb import set_trace as brk

from twitter_data_set import TwitterDataset

# Convolutional neural network (two convolutional layers) #TODO filter_d
class ConvNet(nn.Module):
    def __init__(self, filter_h, out_channels, max_length, filter_d=300, in_channels=1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(filter_h, filter_d)), # check padding in originalcode
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(max_length,1)))
        
    def forward(self, x):
        out = self.layer1(x)
        #out = self.layer2(out)
        #brk()
        out = out.view(out.size(0), -1)
        #out = self.fc(out)
        return out

class CUE_CNN(nn.Module):
    def __init__(self, filters, out_channels, max_length, hidden_units, drop_prob, user_size, num_classes=2):
        super(CUE_CNN, self).__init__()
        self.conv1 = ConvNet(filters[0], out_channels=out_channels, max_length=max_length - filters[0]  + 1)
        self.conv2 = ConvNet(filters[1], out_channels=out_channels, max_length=max_length - filters[1]  + 1)
        self.conv3 = ConvNet(filters[2], out_channels=out_channels, max_length=max_length - filters[2]  + 1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 3 + user_size, hidden_units),
            nn.ReLU(), #dropout
            nn.Dropout(drop_prob),
            nn.Linear(hidden_units, num_classes))
    def forward(self, x, user_embedding):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat((out1, out2, out3, user_embedding), dim=1)
        out = self.fc(out)
        return out

#TODO: FIND APPROPRIATE PARAMS
parser = argparse.ArgumentParser(description='PyTorch CUE CNN Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=128, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0
best_prec1_ps = 0

def plot_stats(epoch, data_1, data_2, data_3, label_1, label_2, label_3,plt):
    plt.plot(range(epoch), data_1, 'r--', label=label_1)
    plt.plot(range(epoch), data_2, 'g--', label=label_2)
    plt.plot(range(epoch), data_3, 'b--', label=label_3)
    plt.legend()


def main():
    run_time = time.ctime().replace(' ', '_')[:-8] 
    directory = 'progress/' + run_time
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(directory + '/logs.txt', 'w')
    global args, best_prec1, best_prec1_ps
    print ("GPU processing available : ", torch.cuda.is_available())
    print ("Number of GPU units available :", torch.cuda.device_count())
    args = parser.parse_args()

    ## READ DATA
    #cudnn.benchmark = True

    #TODO
    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    filter_h = [4,6,8] #[1, 3, 5]
    
    train_sampler = None 
    train_dataset = TwitterDataset(
        csv_file='DATA/txt/bamman_clean.txt', 
        folds_file='DATA/folds/fold_0.csv', 
        word_embedding_file='DATA/embeddings/filtered_embs.txt', 
        user_embedding_file='DATA/embeddings/usr2vec.txt', 
        set_type='train', 
        pad = max(filter_h) - 1
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_dataset = TwitterDataset(
        csv_file='DATA/txt/bamman_clean.txt', 
        folds_file='DATA/folds/fold_0.csv', 
        word_embedding_file='DATA/embeddings/filtered_embs.txt', 
        user_embedding_file='DATA/embeddings/usr2vec.txt', 
        set_type='val', 
        pad = max(filter_h) - 1,
        w2v = train_dataset.w2v
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)
    
    test_dataset = TwitterDataset(
        csv_file='DATA/txt/bamman_clean.txt', 
        folds_file='DATA/folds/fold_0.csv', 
        word_embedding_file='DATA/embeddings/filtered_embs.txt', 
        user_embedding_file='DATA/embeddings/usr2vec.txt', 
        set_type='test', 
        pad = max(filter_h) - 1,
        w2v = train_dataset.w2v
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)
    # epochs
    # print final acc with config
    # print best till date
    for lr in [0.000001, 0.00001]:
        for wd in [1, 0.1, 0.01, 0.001]:
            for oc in [50, 300]:
                for hu in [128, 256]:
                    for dp in [0, 0.1, 0.3]:
                        for fs in [(1,2,3), (1,3,5)]:
                            parameters = {"filters": fs,
                                          "out_channels": oc,                  
                                          "max_length": train_dataset.max_l,
                                          "hidden_units": hu,
                                          "drop_prob": dp,
                                          "user_size": 400,
                                          "epochs":args.epochs}

                            #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                            model = CUE_CNN(parameters['filters'], parameters['out_channels'], parameters['max_length'], parameters['hidden_units'], 
                                            parameters['drop_prob'], parameters['user_size'])#.to(device)
                            model = torch.nn.DataParallel(model).cuda()

                            # define loss function (criterion) and optimizer
                            criterion = nn.CrossEntropyLoss().cuda()

                            optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)
                        #     optimizer = torch.optim.SGD(model.parameters(), lr = args.lr,
                        #                                      momentum=args.momentum,
                        #                                      weight_decay=args.weight_decay)
                        #     torch.optim.Adadelta(model.parameters(), 
                        #                                      rho=args.momentum,
                        #                                      weight_decay=args.weight_decay)

                        #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                            scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, threshold=0.005)
                            # optionally resume from a checkpoint
                            train_prec1_plot = []
                            train_loss_plot = []
                            val_prec1_plot = []
                            val_loss_plot = []
                            test_prec1_plot = []
                            test_loss_plot = []
                            if args.resume:
                                if os.path.isfile(args.resume):
                                    print("=> loading checkpoint '{}'".format(args.resume))
                                    checkpoint = torch.load(args.resume)
                                    args.start_epoch = checkpoint['epoch']
                                    best_prec1 = checkpoint['best_prec1']
                                    model.load_state_dict(checkpoint['state_dict'])
                                    optimizer.load_state_dict(checkpoint['optimizer'])
                                    train_prec1_plot = train_prec1_plot + checkpoint['train_prec1_plot']
                                    train_loss_plot = train_loss_plot + checkpoint['train_loss_plot']
                                    val_prec1_plot = val_prec1_plot + checkpoint['val_prec1_plot']
                                    val_loss_plot = val_loss_plot + checkpoint['val_loss_plot']
                                    test_prec1_plot = test_prec1_plot + checkpoint['test_prec1_plot']
                                    test_loss_plot = test_loss_plot + checkpoint['test_loss_plot']
                                    print("=> loaded checkpoint '{}' (epoch {})"
                                          .format(args.resume, checkpoint['epoch']))
                                else:
                                    print("=> no checkpoint found at '{}'".format(args.resume))

                            if args.evaluate:
                                validate(test_loader, model, criterion, f)
                                return

                            for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
                                #TODO
                                #adjust_learning_rate(optimizer, epoch)

                                # train for one epoch
                                train_prec1, train_loss  = train(train_loader, model, criterion, optimizer, epoch, f)
                                train_prec1_plot.append(train_prec1)
                                train_loss_plot.append(train_loss)

                                # evaluate on validation set
                                val_prec1, val_loss = validate(val_loader, model, criterion, f)
                                val_prec1_plot.append(val_prec1)
                                val_loss_plot.append(val_loss)
                                scheduler.step(val_loss)
                                # evaluate on test set
                                test_prec1,test_loss = validate(test_loader, model, criterion, f, is_val=False)
                                test_prec1_plot.append(test_prec1)
                                test_loss_plot.append(test_loss)

                                # remember best prec@1 and save checkpoint
                                is_best = val_prec1 > best_prec1
                                best_prec1 = max(val_prec1, best_prec1)
                                save_checkpoint({
                                    'train_prec1_plot':train_prec1_plot,
                                    'train_loss_plot':train_loss_plot,
                                    'val_prec1_plot':val_prec1_plot,
                                    'val_loss_plot':val_loss_plot,
                                    'test_prec1_plot':test_prec1_plot,
                                    'test_loss_plot':test_loss_plot,
                                    'epoch': epoch + 1,
                                    'state_dict': model.state_dict(),
                                    'best_prec1': best_prec1,
                                    'optimizer' : optimizer.state_dict(),
                                }, is_best)

                                #plot data
                                #plt.figure(figsize=(12,12))
                                #plt.subplot(2,1,1)
                                #plot_stats(epoch+1, train_loss_plot, val_loss_plot, test_loss_plot, 'train_loss', 'val_loss', 'test_loss', plt)
                                #plt.subplot(2,1,2)
                                #plot_stats(epoch+1, train_prec1_plot, val_prec1_plot, test_prec1_plot, 'train_acc', 'val_acc', 'test_acc', plt)
                                #plt.savefig('progress/' + run_time + '/stats.jpg')
                                #plt.clf()
                                #print "Learning rate is : ", optimizer.param_groups[0]['lr']
                            print " $$ ", lr, wd, oc, hu, dp, fs, " $$ "
                            print train_prec1, val_prec1, test_prec1
                            best_prec1_ps = max(val_prec1, best_prec1_ps)
                            f.write('configuration {0} {1} {2} {3} {4} {5} \n'.format(lr, wd, oc, hu, dp, fs))
                            f.write('train: {0} val: {1} test: {2} \n'.format(train_prec1, val_prec1, test_prec1))
                            f.write('best val performance is : ' + str(best_prec1_ps) + '\n')
                            f.flush()
                            best_prec1_ps = max(val_prec1, best_prec1_ps)
    print "final best performance is for val ", best_prec1_ps                       
    f.close()

def train(train_loader, model, criterion, optimizer, epoch, f):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    #classes = dataset.classes
    for i, (input, user_embeddings, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        #input = input.cuda(async=True)
        input = torch.autograd.Variable(input).type(torch.FloatTensor)
        user_embeddings = torch.autograd.Variable(user_embeddings).type(torch.FloatTensor)
        target = torch.autograd.Variable(target)

        # compute output
        
        output = model(input, user_embeddings)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        #a = list(model.parameters())[0] 
        loss.backward()
        optimizer.step()
        #b = list(model.parameters())[0] 
        #print ("Prahal", torch.equal(a.data, b.data), len(list(model.parameters())))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress_stats = 'Time: {0} Epoch: [{1}][{2}/{3}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   time.ctime()[:-8], epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1)
            #print(progress_stats)
            #f.write(progress_stats + "\n")
            f.flush()
    train_stats = 'Train Time {time} * Prec@1 {top1.avg:.3f} Loss {loss.avg:.4f}'.format(
        time=time.ctime()[:-8],top1=top1, loss=losses)
    #print(train_stats)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, f, is_val=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, user_embeddings, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        #input = input.cuda(async=True)
        input = torch.autograd.Variable(input, volatile=True).type(torch.FloatTensor)
        user_embeddings = torch.autograd.Variable(user_embeddings, volatile=True).type(torch.FloatTensor)
        target = torch.autograd.Variable(target, volatile=True)
        #pdb.set_trace()

        # compute output
        output = model(input, user_embeddings)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1, ))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        #if i % args.print_freq == 0:
#         print('Test: [{0}/{1}]\t'
#               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
#                i, len(val_loader), batch_time=batch_time, loss=losses,
#                top1=top1))

    val_stats = '{is_val} Time {time} * Prec@1 {top1.avg:.3f} Loss {loss.avg:.4f}'.format(
        is_val= 'validation set' if is_val else 'test set',time=time.ctime()[:-8],top1=top1, loss=losses)
    #print(val_stats)
    #f.write(val_stats + "\n")
    #f.flush()
    return top1.avg, losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    target = target.data
    maxk = max(topk)
    batch_size = target.size(0)
    #print ("Prahallllll, output", output.size())
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #print('Correct', correct.size(), target.size(), target.view(1, -1).size(), pred.size())
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    #time.sleep(10)
    return res


if __name__ == '__main__':
    main()