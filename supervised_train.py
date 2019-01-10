from __future__ import print_function
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import *
from logger import *

class Net(nn.Module):
    def __init__(self, in_channel, in_size, struct):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.in_len = in_channel * in_size * in_size #5*10*10
        self.struct = struct
#        self.fc1 = nn.Linear(self.in_len, 512)
#        self.fc2 = nn.Linear(512, 128)
#        self.fc3 = nn.Linear(128, in_size*in_size)
        if struct == 'fc':
            self.net = nn.Sequential(nn.Linear(self.in_len, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, in_size*in_size)
            )
        elif struct == 'conv':
            self.net = nn.Sequential(nn.Conv2d(in_channel, 128,3, padding=1), nn.ReLU(),
            nn.Conv2d(128,128,1), nn.ReLU(),
            nn.Conv2d(128,128,3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 1, 1)
            )


    def forward(self, x):
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        if self.struct == 'fc':
            x = x.view(-1, self.in_len)
        x = self.net(x)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = F.dropout(x, training=self.training)
#        x = self.fc3(x) # normalize the output to be between 0 & 1 ?
#
        return x

def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    f_loss = nn.MSELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device).float()
        optimizer.zero_grad()
        output = model(data)
        loss = f_loss(output, target.view(output.size()))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    writer.scalar_summary('train/loss',loss.item(), epoch)

def test(args, model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    f_loss = nn.MSELoss()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data)
            test_loss += f_loss(output, target.view(output.size()) ).item() # sum up batch loss
            #test_loss += f_loss(output, target.view(output.size(0),-1) ).item() # sum up batch loss
            #pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('                                Test set: Average loss: {:.4f}'.format(test_loss))
    if epoch%100 == 0:
        print(output.view(output.size(0),-1)[0,:25].view(5,5))
    writer.scalar_summary('val/loss',test_loss, epoch)
    #print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
## data generation
    parser.add_argument('--seq-len', type=int, default=20, metavar='N')
    parser.add_argument('--cyc-len', type=int, default=20, metavar='N')
    parser.add_argument('--dm-size', type=int, default=20, metavar='N')
    parser.add_argument('--dataset-size', type=int, default=512, metavar='N')
    parser.add_argument('--gen-rule', type=str, default="gravity-random", metavar='LR')
    #parser.add_argument('--gen-rule', type=str, default="gravity-cycle", metavar='LR')
    #parser.add_argument('--gen-rule', type=str, default="gravity-avg", metavar='LR')
    parser.add_argument('--p', type=float, default=0.5, metavar='LR')
## network structure
    parser.add_argument('--struct', type=str, default="fc", metavar='LR')
    #parser.add_argument('--struct', type=str, default="conv", metavar='LR')

    args = parser.parse_args()

    print(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
                       DMDataset(cyc_len=args.cyc_len,seq_len=args.seq_len, 
        dm_size=args.dm_size, dataset_size=args.dataset_size, 
        gen_rule=args.gen_rule, p=args.p, train=True, seq_num=7),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
                       DMDataset(cyc_len=args.cyc_len,seq_len=args.seq_len, 
        dm_size=args.dm_size, dataset_size=int(args.dataset_size/10), 
        gen_rule=args.gen_rule, p=args.p, train=False, seq_num=3),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


    model = Net(args.seq_len, args.dm_size, args.struct).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    print(model)
    #logdir = "log/"+str(time.time())
    logdir = "log/"+args.gen_rule+'_'+args.struct+str(time.time())
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = Logger(logdir)
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, epoch, writer)


if __name__ == '__main__':
    main()
