import torch.nn as nn
import torch
#from core.math import *
import math


def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)


def normal_log_density(x, mean, log_std, std):
    var = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)


class Value(nn.Module):
    def __init__(self, in_channel, in_size, hidden_size=(128, 128), activation='tanh'):
        super(Value,self).__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        state_dim = in_channel * in_size * in_size #5*10*10

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value

class Policy(nn.Module):
    def __init__(self,in_channel ,in_size, action_dim, hidden_size=(128, 128), activation='tanh', log_std=0):
        super(Policy, self).__init__()
        self.is_disc_action = False
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        state_dim = in_channel * in_size * in_size #5*10*10
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action_mean = nn.Linear(last_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

    def select_action(self, x):
        action_mean, _, action_std = self.forward(x)
        action = torch.normal(action_mean, action_std)
        return action

    def get_kl(self, x):
        mean1, log_std1, std1 = self.forward(x)

        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_mean, action_log_std, action_std = self.forward(x)
        return normal_log_density(actions, action_mean, action_log_std, action_std)

    def get_fim(self, x):
        mean, _, _ = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), mean, {'std_id': std_id, 'std_index': std_index}


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
