from build_model import resnet34
from build_dataset import MOFsDataset
from write_to_log import print_log

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import time
import datetime

from torch.utils.data.distributed import DistributedSampler
import torch.utils.data as Data
import torch.distributed as dist

# local path
# train_data_path = "E:/project/mofs-data/train_h5py.h5"
# test_data_path = 'E:/project/mofs-data/test_h5py.h5'
# 计算平台path
train_data_path = '/data/user10110/mofs-data/train_h5py.h5'
test_data_path = '/data/user10110/mofs-data/test_h5py.h5'
# model save path
model_path = '/home/user10110/code/2_save_model.pkl'
train_png_path = '/home/user10110/code/2_train_loss.png'
test_png_path = '/home/user10110/code/test_loss.png'
rank_path = '/data/user10110/node_rank.txt'
share_path = "/data/user10110/share"

# 设置初始化方式   1: tcp  2: share file
method_flag = 2
# 是否测试标志
test_flag = False

def train(args, device, model, data_loader, epoch, optimizer, loss_func):
    # training
    model.train()
    train_loss = 0
    count = 0
    global x1, y1
    x1 = []
    y1 = []

    print("start trainning!")
    for batch_i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        count += args.batch_size * 1
        optimizer.zero_grad()

        output = model(x)
        # loss
        loss = loss_func(output, y)
        loss.backward()
        optimizer.step()
        # 累加损失值
        train_loss += loss.item()
        train_loss = train_loss / count
        # plt x1,y1
        y1.append(train_loss)
        x1.append(count)

        if (batch_i+1) % args.log_interval == 0:
            print('Epoch:', epoch,
                  '|Step:', batch_i+1,
                  '| loss: %.4f' % train_loss)
    print("trainning finished!")

    if args.rank == 0:
        # painting
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Train Loss')
        plt.plot(x1, y1)
        plt.savefig(train_png_path)
        f = open('loss.txt', 'a')
        for i in range(len(x1)):
            s = 'Step:' + str(x1[i]) + ',' + 'Loss:' + str(format(y1[i], '.4f')) + '\n'
            f.write(s)
        f.close()

def test(args, device, model, test_loader, optimizer, loss_func):
    # testing
    model.eval()    # 让model变成测试模式
    test_loss = 0
    global x2, y2
    x2 = []
    y2 = []
    print("start testing!")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)

            loss = loss_func(output, y)
            test_loss += loss.item()
            test_loss /= (i + 1)
            y2.append(test_loss)
            x2.append(i + 1)

    if args.rank == 0:
        # painting
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Test Loss')
        plt.plot(x2, y2)
        plt.savefig(train_png_path)

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

def new_share_file():
    # 按照时间，新建一个共享文件
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_path = os.path.join(share_path, nowTime)
    # 判断文件夹是否已存在
    isExists = os.path.isfile(file_path)
    if not isExists:
        fp = open(file_path, 'w')
        fp.close()
    # 用一个文件记录共享文件的名称
    record_file_path = os.path.join(share_path, 'record_file_path.txt')
    fp = open(record_file_path, 'w')
    fp.write(file_path)
    fp.close()

def set_init_method(method_flag):
    if method_flag == 1:    # tcp
        ip = "192.168.101.40"
        port = "23456"
        init_method = "tcp://" + ip + ":" + port
    else:                   # shared file
        record_file_path = os.path.join(share_path, 'record_file_path.txt')
        fp = open(record_file_path, 'r')
        file_path = fp.readline()
        fp.close()
        init_method = "file://" + file_path
    return init_method

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Distributed trainning')
    parser.add_argument('--data-size', type=int, default=55000, metavar='N',
                        help='number of train-data for training')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training ')
    parser.add_argument('--test-data-size', type=int, default=10, metavar='N',
                        help='number of test-data for testing')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing ')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    # init method: tcp://192.168.101.40:23456
    # file:///data/user10110/share
    parser.add_argument('--init-method', type=str, default='tcp://192.168.101.40:23456')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=3)

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 设置每台机器的rank
    set_rank(args)
    #  设置进程初始化方式 init_method  1: tcp  2: file
    if method_flag != 1:
        if args.rank == 0:
            new_share_file()
    args.init_method = set_init_method(method_flag)
    print(args)

    # 1、初始化进程组
    dist.init_process_group(backend="gloo",
                            init_method=args.init_method,
                            timeout=datetime.timedelta(days=1),
                            world_size=args.world_size,
                            rank=args.rank)

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    # 2、分发数据DistributedSampler
    print("start loading data!")
    train_data = MOFsDataset(data_size=args.data_size, file_path=train_data_path)
    train_sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=args.rank)  # world_size总进程数，rank进程序号
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch_size, sampler=train_sampler)  # shuffle随机化

    test_data = MOFsDataset(data_size=args.test_data_size, file_path=test_data_path)
    test_sampler = DistributedSampler(test_data, num_replicas=args.world_size, rank=args.rank)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=args.test_batch_size, shuffle=False, sampler=test_sampler)  # shuffle随机化

    # 3、DDP模型
    model = resnet34().to(device)
    if use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            model) if use_cuda else torch.nn.parallel.DistributedDataParallelCPU(model)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # loss_func = nn.MSELoss()
    loss_func = nn.SmoothL1Loss()

    # 如果test_flag=True,则加载已保存的模型
    if test_flag:
        # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        test(args, device, model, test_loader, optimizer, loss_func)
        return

    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('Have load epoch {} ！'.format(start_epoch))
    else:
        start_epoch = 1
        print('no model,trainning from epoch 1 !')

    total_time = 0
    for epoch in range(start_epoch, args.epochs+start_epoch):
        # 设置epoch位置，这应该是个为了同步所做的工作
        train_sampler.set_epoch(epoch)
        # test_sampler.set_epoch(epoch)

        # 开始trainning
        start_cpu_secs = time.time()
        # trainning
        train(args, device, model, train_loader, epoch, optimizer, loss_func)
        end_cpu_secs = time.time()
        total_time += end_cpu_secs - start_cpu_secs
        print("Epoch {} took {:.3f} minutes".format(
            epoch, (end_cpu_secs - start_cpu_secs) / 60))

        # 开始testing
        # test(args, device, model, test_loader, optimizer, loss_func)

        # 保存模型
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}
        if args.rank == 0:
            torch.save(state, model_path)

    print("Total trainning time= {:.3f} minutes".format(total_time / 60))
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
    print_log()

