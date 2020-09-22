import os
import random
import shutil

train_data_dir = "E:/project/mofs-data/later_data/train_data"
test_data_dir = "E:/project/mofs-data/later_data/test_data"
dirs = ['data_less_than_60/', 'data_60_to_120/', 'data_120_to_180/', 'data_180_to_240/', 'data_240_to_300/',
            'data_300_to_360/', 'data_360_to_420/', 'data_more_than_420/']

def split_data():
    # 每个范围的吸附值，数据取20%，划分为测试集
    test_num = 0
    for dir in dirs:
        train_range = os.path.join(train_data_dir, dir)
        test_range = os.path.join(test_data_dir, dir)
        if not os.path.isdir(test_range):
            os.makedirs(test_range)

        files = os.listdir(train_range)  # 训练数据集中的每一个cif文件
        filenumber = len(files)
        rate = 0.20
        picknumber = int(filenumber * rate)
        test_files = random.sample(files, picknumber)   # 从训练集中随机选取百分之20，作为测试集
        for file in test_files:
                shutil.move(train_range + file, test_range + file)
        test_num += picknumber

    print(test_num)

def main():
    split_data()
    print("Have Split Data")

if  __name__ == '__main__':
    main()