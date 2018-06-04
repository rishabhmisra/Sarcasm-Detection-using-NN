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
from pdb import set_trace as brk
import time

from pdb import set_trace as brk

# from SemEval_data_set import SemEvalDataset
from headline_data_set import HeadlineDataset

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
        out = out.view(out.size(0), -1)
        return out
    
class CUE_CNN(nn.Module):
    def __init__(self, filters, out_channels, max_length, hidden_units, drop_prob, num_classes=2):
        super(CUE_CNN, self).__init__()
        self.conv1 = ConvNet(filters[0], out_channels=out_channels, max_length=max_length - filters[0]  + 1)
        self.conv2 = ConvNet(filters[1], out_channels=out_channels, max_length=max_length - filters[1]  + 1)
        self.conv3 = ConvNet(filters[2], out_channels=out_channels, max_length=max_length - filters[2]  + 1)
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat((out1, out2, out3), dim=1)
        return out

class MixtureOfExperts(nn.Module):
    def __init__(self, filters, out_channels, max_length, hidden_units, drop_prob, lstm_input_size, hidden_size_lstm, hidden_units_attention, pretrained_weight, num_classes=2):
        super(MixtureOfExperts, self).__init__()
        #brk()
        self.embed = nn.Embedding(pretrained_weight.shape[0], pretrained_weight.shape[1])
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        
        self.cue_cnn = CUE_CNN(filters, out_channels, max_length, hidden_units, drop_prob, num_classes)
        self.bi_lstm = nn.LSTM(lstm_input_size, hidden_size_lstm, num_layers=1, bidirectional=True)
        
#         self.attention_mlp = nn.Sequential(
#             nn.Linear(hidden_size_lstm * 2, hidden_units_attention),
#             nn.ReLU(),
#             nn.Linear(hidden_units_attention, 1))
        self.attention_mlp = nn.Linear(hidden_size_lstm * 2, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(out_channels * 3 + hidden_size_lstm * 2, hidden_units),
            nn.Tanh(), 
            nn.Dropout(drop_prob), #dropout
            nn.Linear(hidden_units, 50),
            nn.Tanh(), 
            nn.Dropout(drop_prob), #dropout
            nn.Linear(50, num_classes))
    
    def forward(self, x):
        x = self.embed(x)
        out1 = self.cue_cnn(x.unsqueeze(1))
        out2 = self.bi_lstm(x.transpose(0,1))[0].transpose(0,1)
        out3 = self.attention_mlp(out2)
        #brk()
        out4 = torch.mul(nn.Softmax()(out3.view(x.size(0), x.size(1))).unsqueeze(2).repeat(1,1,out2.size(2)), out2)
        out5 = torch.sum(out4, dim=1)
        out = torch.cat((out1, out5), dim=1)
        out = self.mlp(out)
        return out
    

#TODO: FIND APPROPRIATE PARAMS
parser = argparse.ArgumentParser(description='PyTorch CUE CNN Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.95, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=128, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0
best_prec1_ps = 0

def plot_stats(epoch, data_1, data_2, data_3, label_1, label_2, label_3, plt):
    plt.plot(range(epoch), data_1, 'r--', label=label_1)
    plt.plot(range(epoch), data_2, 'g--', label=label_2)
    plt.plot(range(epoch), data_3, 'b--', label=label_3)
    #plt.plot(range(epoch), data_4, 'y--', label=label_4)
    plt.legend()


def main():
    run_time = time.ctime().replace(' ', '_')[:-8] 
    directory = 'progress/' + run_time
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(directory + '/logs.txt', 'w',encoding='utf-8')
    f1 = open(directory + '/vis.txt', 'w',encoding='utf-8')
    global args, best_prec1, best_prec1_ps
    print ("GPU processing available : ", torch.cuda.is_available())
    print ("Number of GPU units available :", torch.cuda.device_count())
    args = parser.parse_args()

    ## READ DATA
    filter_h = [4,6,8] #[1, 3, 5]
    
    train_sampler = None 
    train_dataset = HeadlineDataset(
        csv_file='DATA/txt/headline_train.txt', 
        #folds_file='DATA/folds/fold_0.csv', 
        word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
        #user_embedding_file='DATA/embeddings/usr2vec.txt', 
        #set_type='train', 
        pad = max(filter_h) - 1,
        whole_data='DATA/txt/headlines_clean.txt',
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_dataset = HeadlineDataset(
        csv_file='DATA/txt/headline_val.txt', 
        #folds_file='DATA/folds/fold_0.csv', 
        word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
        #user_embedding_file='DATA/embeddings/usr2vec.txt', 
        #set_type='val', 
        pad = max(filter_h) - 1,
        word_idx = train_dataset.word_idx,
        pretrained_embs = train_dataset.pretrained_embs,
        max_l=train_dataset.max_l,
        #w2v = train_dataset.w2v
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True)
    
    test_dataset = HeadlineDataset(
        csv_file='DATA/txt/headline_test.txt', 
        #folds_file='DATA/folds/fold_0.csv', 
        word_embedding_file='DATA/embeddings/headlines_filtered_embs.txt', 
        #user_embedding_file='DATA/embeddings/usr2vec.txt', 
        #set_type='test', 
        pad = max(filter_h) - 1,
        word_idx = train_dataset.word_idx,
        pretrained_embs = train_dataset.pretrained_embs,
        #w2v = train_dataset.w2v
        max_l=train_dataset.max_l,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=None,
        num_workers=args.workers, pin_memory=True)

    
#     semeval_dataset = SemEvalDataset(
#         csv_file='DATA/txt/SemEval_clean.txt', 
#         word_embedding_file='DATA/embeddings/SemEval_filtered_embs.txt',  
#         pad = max(filter_h) - 1,
#         max_l = train_dataset.max_l,
#         #word_idx = train_dataset.word_idx,
#         #pretrained_embs = train_dataset.pretrained_embs,
#     )

#     semeval_loader = torch.utils.data.DataLoader(
#         semeval_dataset, batch_size=args.batch_size, shuffle=None,
#         num_workers=args.workers, pin_memory=True)
    for lr in [0.05]:
        for wd in [0.01]:
            for oc in [200]:
                for hu in [100]:
                    for dp in [0.2]:
                        for hsl in [128]:
                            for fs in [(4,6,8)]:
                                best_prec1 = 0
                                best_prec1_index = -1
                                parameters = {"filters": fs,
                                              "out_channels": oc,                  
                                              "max_length": train_dataset.max_l + 2  * (max(filter_h) - 1),
                                              "hidden_units": hu,
                                              "drop_prob": dp,
                                              "user_size": 400,
                                              "epochs":args.epochs,
                                              "hidden_size_lstm":hsl,
                                              "wd":wd}

                                #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                                model = MixtureOfExperts(parameters['filters'], parameters['out_channels'], parameters['max_length'], parameters['hidden_units'], parameters['drop_prob'], 300, parameters['hidden_size_lstm'], 128, train_dataset.pretrained_embs)
                                model = torch.nn.DataParallel(model).cuda()

                                # define loss function (criterion) and optimizer
                                criterion = nn.CrossEntropyLoss().cuda()

                            #     optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
                            #     optimizer = torch.optim.SGD(model.parameters(), lr = args.lr,
                            #                                      momentum=args.momentum,
                            #                                      weight_decay=args.weight_decay)
                                optimizer = torch.optim.Adadelta(model.parameters(), lr = lr,
                                                                 rho=args.momentum,
                                                                 weight_decay=wd)

                                # optionally resume from a checkpoint
                                train_prec1_plot = []
                                train_loss_plot = []
                                val_prec1_plot = []
                                val_loss_plot = []
                                test_prec1_plot = []
                                test_loss_plot = []
                            #     semeval_prec1_plot = []
                            #     semeval_loss_plot = []
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
                            #             semeval_prec1_plot = semeval_prec1_plot + checkpoint['semeval_prec1_plot']
                            #             semeval_loss_plot = semeval_loss_plot + checkpoint['semeval_loss_plot']
                                        print("=> loaded checkpoint '{}' (epoch {})"
                                              .format(args.resume, checkpoint['epoch']))
                                    else:
                                        print("=> no checkpoint found at '{}'".format(args.resume))

                                if args.evaluate:
                                    validate(semeval_loader, model, criterion, f, f1, tag='semeval')
                                    return

                                for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
                                    adjust_learning_rate(optimizer, epoch, lr)
                                    # train for one epoch
                                    train_prec1, train_loss  = train(train_loader, model, criterion, optimizer, epoch, f)
                                    train_prec1_plot.append(train_prec1)
                                    train_loss_plot.append(train_loss)

                                    # evaluate on validation set
                                    val_prec1, val_loss = validate(val_loader, model, criterion, f, f1, tag='val')
                                    val_prec1_plot.append(val_prec1)
                                    val_loss_plot.append(val_loss)

                                    # evaluate on test set
                                    test_prec1,test_loss = validate(test_loader, model, criterion, f, f1, tag='test')
                                    test_prec1_plot.append(test_prec1)
                                    test_loss_plot.append(test_loss)

                                    # evaluate on semeval set
                            #         semeval_prec1,semeval_loss = validate(semeval_loader, model, criterion, f, f1, tag='semeval')
                            #         semeval_prec1_plot.append(semeval_prec1)
                            #         semeval_loss_plot.append(semeval_loss)
                                    f1.close()
                                    # remember best prec@1 and save checkpoint
                                    val_prec1 = val_prec1.cpu()
                                    is_best = val_prec1 > best_prec1
                                    #brk()
                                    best_prec1 = max(val_prec1, best_prec1)
                                    save_checkpoint({
                                        'train_prec1_plot':train_prec1_plot,
                                        'train_loss_plot':train_loss_plot,
                                        'val_prec1_plot':val_prec1_plot,
                                        'val_loss_plot':val_loss_plot,
                                        'test_prec1_plot':test_prec1_plot,
                                        'test_loss_plot':test_loss_plot,
                            #             'semeval_prec1_plot':test_prec1_plot,
                            #             'semeval_loss_plot':test_loss_plot,
                                        'epoch': epoch + 1,
                                        'state_dict': model.state_dict(),
                                        'best_prec1': best_prec1,
                                        'optimizer' : optimizer.state_dict(),
                                    }, is_best)

                                    #plot data
    #                                 plt.figure(figsize=(12,12))
    #                                 plt.subplot(2,1,1)
    #                                 plot_stats(epoch+1, train_loss_plot, val_loss_plot, test_loss_plot, 'train_loss', 'val_loss', 'test_loss', plt)
    #                                 plt.subplot(2,1,2)
    #                                 plot_stats(epoch+1, train_prec1_plot, val_prec1_plot, test_prec1_plot, 'train_acc', 'val_acc', 'test_acc', plt)
    #                                 plt.savefig('progress/' + run_time + '/stats.jpg')
    #                                 plt.clf()
                                print (" $$ ", lr, wd, oc, hu, hsl, dp, fs, " $$ ")
                                #brk()
                                print (train_prec1_plot[best_prec1_index], best_prec1, test_prec1_plot[best_prec1_index])
                                best_prec1_ps = max(best_prec1, best_prec1_ps)
                                f.write('configuration {0} {1} {2} {3} {4} {5} {6} \n'.format(lr, wd, oc, hu, hsl, dp, fs))
                                f.write('train: {0} val: {1} test: {2} \n'.format(train_prec1_plot[best_prec1_index], best_prec1, test_prec1_plot[best_prec1_index]))
                                f.write('best val performance is : ' + str(best_prec1_ps) + '\n')
                                f.flush()
    print ("final best performance is for val ", best_prec1_ps)                       
    f.close()
    #f1.close()

def train(train_loader, model, criterion, optimizer, epoch, f):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, sent) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input = input.cuda(async=True)
        input = torch.autograd.Variable(input).type(torch.LongTensor)
        #user_embeddings = torch.zeros(input.size(0), 400)
        #user_embeddings = torch.autograd.Variable(user_embeddings).type(torch.FloatTensor)
        target = torch.autograd.Variable(target)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, f, f1, tag):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data_points in enumerate(val_loader):
        if tag != 'semeval':
            input, target, sent = data_points
        else:
            input, user_embeddings, target, sents = data_points
        target = target.cuda(async=True)
        input = input.cuda(async=True)
        input = torch.autograd.Variable(input, volatile=True).type(torch.LongTensor)
#         user_embeddings = torch.zeros(input.size(0), 400)
#         user_embeddings = torch.autograd.Variable(user_embeddings, volatile=True).type(torch.FloatTensor)
        target = torch.autograd.Variable(target, volatile=True)
        #pdb.set_trace()
        #print(sent)
        # compute output
        output = model(input)
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
        
        if tag == 'semeval' and args.evaluate:
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            for x in range(target.size(0)):
                progress_stats = '{label} | {pred} | {sent} \n'.format(label=target[x].data[0],pred=pred[0][x].data[0],
                   sent=sents[x])
                #print(progress_stats)
                f1.write(progress_stats)
            f1.flush()
    #brk()
    val_stats = '{tag}: Time {time} * Prec@1 {top1.avg:.3f} Loss {loss.avg:.4f}'.format(
        tag=tag,time=time.ctime()[:-8],top1=top1, loss=losses)
    print(val_stats)
    #f.write(val_stats + "\n")
    f.flush()
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


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.8** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    target = target.data
    maxk = max(topk)
    batch_size = target.size(0)
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