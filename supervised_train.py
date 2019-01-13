from __future__ import print_function
import os
import sys
import time
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import *
from logger import *
from core.models import *
from core.common import * 

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

dtype = torch.float64

def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    f_loss = nn.MSELoss()
    agg_loss = 0
    for batch_idx, (data, target,_) in enumerate(train_loader):
        data, target = data.to(device).to(dtype), target.to(device).to(dtype)
        optimizer.zero_grad()
        output = model(data)
        loss = f_loss(output, target.view(output.size()))
        loss.backward()
        optimizer.step()
        agg_loss += loss.item()
        print(len(train_loader),len(train_loader.dataset))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    writer.scalar_summary('train/loss',agg_loss/len(train_loader.dataset), epoch)

def test(args, model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    f_loss = nn.MSELoss()
    correct = 0
    with torch.no_grad():
        for data, target,_ in test_loader:
            data, target = data.to(device).to(dtype), target.to(device).to(dtype)
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

def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, clip_epsilon, l2_reg):
    value_net.train()
    policy_net.train()
    """update critic"""
    #for _ in range(optim_value_iternum):
    values_pred = value_net(states)
    value_loss = (values_pred - returns).pow(2).mean()
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    log_probs = policy_net.get_log_prob(states, actions)
    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()
    return value_loss.item(), policy_surr.item()

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
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--approach', type=str, default="supervised", metavar='LR')

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
# PPO arguments
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
args = parser.parse_args()

print(args)

use_cuda = not args.no_cuda and torch.cuda.is_available()

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)

torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

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


#logdir = "log/"+str(time.time())
logdir = "log/"+args.gen_rule+'_'+args.struct+str(time.time())
if not os.path.exists(logdir):
    os.makedirs(logdir)
writer = Logger(logdir)

def main():
  if args.approach == "supervised":
    model = Net(args.seq_len, args.dm_size, args.struct).to(device)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, epoch, writer)

  elif args.approach == "ppo":
    """environment"""
    state_dim = args.dm_size**2
    #running_state = ZFilter((state_dim,), clip=5)
    
    """define actor and critic"""
    args.edge_num = 10
    if args.model_path is None:
        policy_net = Policy(args.seq_len, args.dm_size, args.edge_num, log_std=args.log_std)
        value_net = Value(args.seq_len, args.dm_size)
    else:
        policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
    policy_net.to(device)
    value_net.to(device)
    
    optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.lr)

    Rmax, Rmin, Ravg = 0, 0, 0
    for epoch in range(1, args.epochs + 1):
        agg_loss_v, agg_loss_p = 0, 0
        for batch_idx, (states, actions, rewards, masks) in enumerate(train_loader):
            t0 = time.time()
            states = states.to(device).to(dtype)
            actions = actions.to(device).to(dtype)
            rewards = rewards.to(device).to(dtype)
            masks = masks.to(device).to(dtype)
            with torch.no_grad():
                values = value_net(states)
                fixed_log_probs = policy_net.get_log_prob(states, actions)

            """get advantage estimation from the trajectories"""
            advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)
            loss_v, loss_p = ppo_step(policy_net, value_net, optimizer_policy,\
                    optimizer_value, 1, states, actions, returns,\
                    advantages, fixed_log_probs, args.clip_epsilon, args.l2_reg)
            agg_loss_v += loss_v.item()
            agg_loss_p += loss_p.item()

            """perform mini-batch PPO update"""
            if batch_idx % args.log_interval == 0:
                t1 = time.time()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_v/p: {:.4f}/{:.4f}, Rmax/min/avg: {}/{}/{}, time: {:.3f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_v, loss_p, Rmax, Rmin, Ravg, t1-t0))
        """clean up gpu memory"""
        torch.cuda.empty_cache()
        writer.scalar_summary('train/loss_v',agg_loss_v/len(train_loader.dataset), epoch)
        writer.scalar_summary('train/loss_p',agg_loss_p/len(train_loader.dataset), epoch)

if __name__ == '__main__':
    main()
