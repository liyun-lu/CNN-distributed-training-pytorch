from build_model import resnet34
from build_dataset import MOFsDataset

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time

#超参数
EPOCHS = 1
BATCH_SIZE = 1000
TRAIN_DATA_SIZE = 55000
TEST_DATA_SIZE = 1
LR = 0.001
TNTERVAL = 1   #输出间隔
test_flag = False  #测试标志，True时加载保存好的模型进行测试

# local path
# train_data_path = "E:/project/mofs-data/test_h5py.h5"
# test_data_path = 'E:/project/mofs-data/test_h5py.h5'
# 计算平台path
train_data_path = '/data/user10110/mofs-data/train_h5py.h5'
test_data_path = '/data/user10110/mofs-data/test_h5py.h5'
model_path = 'one_save_model.pkl'  # 模型保存路径


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("start loading data! Batch size is: " + str(BATCH_SIZE))
train_data = MOFsDataset(data_size=TRAIN_DATA_SIZE, file_path=train_data_path)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  # shuffle随机化
test_data = MOFsDataset(data_size=TEST_DATA_SIZE, file_path=test_data_path)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)  # shuffle随机化

model = resnet34()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.SmoothL1Loss()

def train(model, data_loader, epoch):
    # training
    model.train()
    train_loss = 0
    count = 0
    global x1, y1
    x1 = []
    y1 = []

    print("start trainning!")
    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        count += 1 * BATCH_SIZE
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

        if (i+1) % TNTERVAL == 0:
            print('Epoch:', epoch,
                  '| Step:', i+1,
                  '| loss: %.4f' % train_loss)
    print("trainning finished!")
    # painting
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.plot(x1, y1)
    plt.savefig('one_train_loss.png')
    f = open('one_loss.txt', 'a')
    for i in range(len(x1)):
        s = 'Step:' + str(x1[i]) + ',' + 'Loss:' + str(format(y1[i], '.4f')) + '\n'
        f.write(s)
    f.close()

def test(model, test_loader):
    # testing
    model.eval()    # 让model变成测试模式
    test_loss = 0
    test_loss_y = 0
    correct = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            optimizer.zero_grad()
            output = model(x)

            loss = loss_func(output, y)
            test_loss += loss.item()

            # _, pred = torch.max(output.data, 1)
            # print(float(y.numpy()), float(pred.numpy()))
            # total += y.size(0)
            # correct += (abs(pred - y) < 1).sum()
            loss_y = abs(output - y)
            loss_y = float(loss_y.data.cpu().numpy())  # tensor -> numpy
            test_loss_y += loss_y
            test_loss_y = test_loss_y / (i+1)
            if test_loss_y < 1:
                correct += 1

        test_loss /= (i + 1)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    #如果test_flag=True,则加载已保存的模型
    if test_flag:
        # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        test(model, test_loader)
        return

    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print('load the epoch {} '.format(start_epoch))
    else:
        start_epoch = 1
        print('no model,trainning from epoch 1 !')

    total_time = 0
    for epoch in range(1, EPOCHS+1):
        start_cpu_secs = time.time()

        train(model, train_loader, epoch)
        # test(model, test_loader)

        end_cpu_secs = time.time()
        total_time += end_cpu_secs - start_cpu_secs
        print("Epoch {} of {} took {:.3f} minutes".format(
            epoch, EPOCHS, (end_cpu_secs - start_cpu_secs) / 60))

        #保存模型
        state = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch}
        torch.save(state, model_path)
    print("Total time= {:.3f} minutes".format(total_time / 60))


if __name__ == '__main__':
    main()
