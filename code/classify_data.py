import os
import pandas as pd
import shutil
import random

original_dir = "E:/project/mofs-data/Original_GA_MOFs"
train_data_dir = "E:/project/mofs-data/later_data/train_data"
test_data_dir = "E:/project/mofs-data/later_data/test_data"
gene_uptake_path = "E:/project/mofs-data/gene_uptake.xlsx"

def clssify_data():
    num = 0
    sub_dirs = os.listdir(original_dir)  # 所有mof.cif文件构成一个list
    struct_vals = pd.DataFrame(pd.read_excel(gene_uptake_path))  # 读取xlsx文件中原子结构的吸附值
    for sub_dir in sub_dirs:
        struct = sub_dir.split(".")[0]
        struct = struct[struct.find("_", struct.find("_") + 1) + 1:]  # 原子结构
        val = struct_vals.query("gene_structure == '" + struct + "'")["uptake(CH4)"].astype(float)  # 吸附值
        value = float(val.values)
        sub_dir = os.path.join(original_dir, sub_dir)  # 每一个cif文件
        # 根据吸附值分类cif文件
        if value < 60:
            data_range = os.path.join(train_data_dir, "data_less_than_60")
            if not os.path.isdir(data_range):
                os.makedirs(data_range)
            shutil.move(sub_dir, data_range)    # 将吸附值响应的cif文件移到新建的文件夹中
        elif 60 < value < 120:
            data_range = os.path.join(train_data_dir, "data_60_to_120")
            if not os.path.isdir(data_range):
                os.makedirs(data_range)
            shutil.move(sub_dir, data_range)
        elif 120 < value < 180:
            data_range = os.path.join(train_data_dir, "data_120_to_180")
            if not os.path.isdir(data_range):
                os.makedirs(data_range)
            shutil.move(sub_dir, data_range)
        elif 180 < value < 240:
            data_range = os.path.join(train_data_dir, "data_180_to_240")
            if not os.path.isdir(data_range):
                os.makedirs(data_range)
            shutil.move(sub_dir, data_range)
        elif 240 < value < 300:
            data_range = os.path.join(train_data_dir, "data_240_to_300")
            if not os.path.isdir(data_range):
                os.makedirs(data_range)
            shutil.move(sub_dir, data_range)
        elif 300 < value < 360:
            data_range = os.path.join(train_data_dir, "data_300_to_360")
            if not os.path.isdir(data_range):
                os.makedirs(data_range)
            shutil.move(sub_dir, data_range)
        elif 360 < value < 420:
            data_range = os.path.join(train_data_dir, "data_360_to_420")
            if not os.path.isdir(data_range):
                os.makedirs(data_range)
            shutil.move(sub_dir, data_range)
        else:
            data_range = os.path.join(train_data_dir, "data_more_than_420")
            if not os.path.isdir(data_range):
                os.makedirs(data_range)
            shutil.move(sub_dir, data_range)

        num += 1
    print(num)

def main():
    clssify_data()  # 51163
    print("Have Clssify Data")

if  __name__ == '__main__':
    main()