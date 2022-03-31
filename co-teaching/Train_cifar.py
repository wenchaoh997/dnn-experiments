import os
import sys
sys.path.insert(0, './')

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import random
import argparse

# others
from nets.PreResNet import *
import dataloader_cifar as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=16, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--nr', default=0.5, type=float, help='noise ratio')
parser.add_argument('--f', default=0.5, type=float, help='forget ratio')
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=924)
parser.add_argument('--UseCUDA', default=True, type=bool)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='../_datasets/cifar10/cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

if args.UseCUDA and torch.cuda.is_available():
    torch.cuda.set_device(args.gpuid)
    DEVICE = torch.device("cuda")
    cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# define drop rate schedule
rate_schedule = np.ones(args.num_epochs)*args.f
rate_schedule[:args.num_gradual] = np.linspace(0, args.f**args.exponent, args.num_gradual)

# official loss
def loss_coteaching(y_1, y_2, t, forget_rate, ind):
    """
    y_1: soft-prediction from net1
    y_2: soft-prediction from net2
    t: targets
    forget_rate: current forget rate
    ind: raw index in training set

    noise_or_not: masking real labels, if same marking True, else False
    """
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data.cpu()).to(DEVICE)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data.cpu()).to(DEVICE)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2


def train_one_epoch(epoch, net, net2, optimizer, optimizer2, dataloader):
    net.train()
    net2.train()
    train_loss_1 = 0
    train_loss_2 = 0
    pure_ratio_1_list=[]
    pure_ratio_2_list=[]
    
    num_iter = ( len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, index) in enumerate(dataloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        logits1 = net(inputs)
        logits2 = net2(inputs)

        # co-teaching loss
        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(logits1, logits2, \
            labels, rate_schedule[epoch], index)
        pure_ratio_1_list.append(100*pure_ratio_1)
        pure_ratio_2_list.append(100*pure_ratio_2)
        train_loss_1 += loss_1.cpu()
        train_loss_2 += loss_2.cpu()
        # update model_1
        optimizer.zero_grad()
        loss_1.backward()
        optimizer.step()
        # update model_2
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        sys.stdout.write("\r")
        sys.stdout.write('%s : %.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t loss-1: %.4f | loss-2: %.4f'
                %(args.dataset, args.nr, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, train_loss_1/(batch_idx+1), train_loss_2/(batch_idx+1)))
        sys.stdout.flush()
    # save results
    mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
    mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
    print('\tPure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (mean_pure_ratio1, mean_pure_ratio2))
    test_log.write("Epoch #%d\tPure_Ratio-1: %.4f %%, Pure_Ratio-2: %.4f %% \n"\
                    %(epoch, mean_pure_ratio1, mean_pure_ratio2))
    test_log.flush()


def test_one_epoch(epoch, net, dataloader, record):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = net(inputs)
            # return max_value, index
            _, prediction = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += prediction.eq(labels).cpu().sum().item()
    acc = 100. * correct/total
    print("\n[INFO] Test Epoch #%d\t Accuracy: %.2f%%\n"%(epoch,acc))
    test_log.write("Epoch #%d\t Accuracy: %.2f%%\n"%(epoch,acc))
    test_log.flush()
    record.append(acc)
    return record

def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.to(DEVICE)
    return model

if __name__=="__main__":
    try:
        os.mkdir("./co-teaching/checkpoint/")
    except:
        pass

    stats_log=open('./co-teaching/checkpoint/%s_%.1f_%s'%(args.dataset,args.nr,args.noise_mode)+'_stats.txt','w')
    test_log=open('./co-teaching/checkpoint/%s_%.1f_%s'%(args.dataset,args.nr,args.noise_mode)+'_acc.txt','w')

    loader = dataloader.cifar_dataloader(
        args.dataset, r=args.nr, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=0, \
        root_dir=args.data_path, log=stats_log, noise_file='%s/%.1f_%s.json'%(args.data_path,args.nr,args.noise_mode) )

    print("[INFO] Building net.\n")
    net1 = create_model()
    net2 = create_model()

    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    all_loss = [[], []]
    acc_record = []

    train_loader = loader.run("warmup")
    eval_loader = loader.run("eval_train")
    test_loader = loader.run("test")

    # masking noise_or_not
    clean_train_label = json.load( \
            open("%s/%.1f_%s_clean.json"%(args.data_path,args.nr,args.noise_mode),"r"))
    curr_train_label = json.load( \
            open("%s/%.1f_%s.json"%(args.data_path,args.nr,args.noise_mode),"r"))
    noise_or_not = np.transpose(curr_train_label)==np.transpose(clean_train_label)

    for epoch in range(args.num_epochs + 1):
        lr = args.lr
        if epoch >= 150:
            lr /= 10

        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr
        for param_group in optimizer2.param_groups:
            param_group["lr"] = lr

        train_one_epoch(epoch, net1, net2, optimizer1, optimizer2, train_loader)
        acc_record = test_one_epoch(epoch, net1, test_loader, acc_record)

