import os
import sys
sys.path.insert(0, './')

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
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--nr', default=0.5, type=float, help='noise ratio')
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

def train_one_epoch(epoch, net, optimizer, dataloader):
    net.train()
    train_loss_1 = 0

    num_iter = ( len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_1 += loss.cpu()

        sys.stdout.write("\r")
        sys.stdout.write('%s : %.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.nr, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, 
                train_loss_1/(batch_idx+1)))
        sys.stdout.flush()

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
        os.mkdir("./CE/checkpoint/")
    except:
        pass

    stats_log=open('./CE/checkpoint/%s_%.1f_%s'%(args.dataset,args.nr,args.noise_mode)+'_stats.txt','w')
    test_log=open('./CE/checkpoint/%s_%.1f_%s'%(args.dataset,args.nr,args.noise_mode)+'_acc.txt','w')

    loader = dataloader.cifar_dataloader(
        args.dataset, r=args.nr, noise_mode=args.noise_mode, batch_size=args.batch_size, num_workers=0, \
        root_dir=args.data_path, log=stats_log, noise_file='%s/%.1f_%s.json'%(args.data_path,args.nr,args.noise_mode) )

    print("[INFO] Building net.\n")
    net1 = create_model()

    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    all_loss = [[], []]
    acc_record = []

    train_loader = loader.run("warmup")
    eval_loader = loader.run("eval_train")
    test_loader = loader.run("test")

    for epoch in range(args.num_epochs + 1):
        lr = args.lr
        if epoch >= 150:
            lr /= 10

        for param_group in optimizer1.param_groups:
            param_group["lr"] = lr

        train_one_epoch(epoch, net1, optimizer1, train_loader)
        acc_record = test_one_epoch(epoch, net1, test_loader, acc_record)

