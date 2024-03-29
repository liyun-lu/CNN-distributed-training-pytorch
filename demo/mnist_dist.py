from __future__ import print_function
import argparse
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

rank_path = '/data/user10110/node_rank.txt'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def set_rank(args):
    fp = open(rank_path, "r+")
    args.rank = int(fp.read())
    fp.seek(0)     # 定位到position 0
    fp.truncate()  # 清空文件
    if args.rank+1 == args.world_size:
        fp.write("0")
    else:
        fp.write(str(args.rank+1))
    fp.close()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--init-method', type=str, default='tcp://192.168.101.43:23456')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # 获取每台机器的rank
    set_rank(args)
    print(args)

    # 1、初始化
    dist.init_process_group(backend="gloo",
                            init_method=args.init_method,
                            timeout=datetime.timedelta(days=1),
                            world_size=args.world_size,
                            rank=args.rank)

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    path = '/data/user10110/mnist'
    train_dataset = datasets.MNIST(path, train=True, download=False,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    # 2、分发数据
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(path, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # 3、DDL
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    model = Net().to(device)
    if use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            model) if use_cuda else torch.nn.parallel.DistributedDataParallelCPU(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    total_time = 0
    for epoch in range(1, args.epochs + 1):
        # 设置epoch位置，这应该是个为了同步所做的工作
        train_sampler.set_epoch(epoch)

        start_cpu_secs = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        end_cpu_secs = time.time()

        print("Epoch {} of {} took {:.3f}s".format(
            epoch, args.epochs, end_cpu_secs - start_cpu_secs))
        total_time += end_cpu_secs - start_cpu_secs

        test(args, model, device, test_loader)

    print("Total time= {:.3f}s".format(total_time))


if __name__ == '__main__':
    main()
