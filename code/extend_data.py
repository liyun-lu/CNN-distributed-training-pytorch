import os
import numpy as np
import random
#import math

file_dir = "E:/project/mofs-data/later_data/train_data"
dirs = ['data_less_than_60/', 'data_60_to_120/', 'data_120_to_180/', 'data_180_to_240/', 'data_240_to_300/',
            'data_300_to_360/', 'data_360_to_420/', 'data_more_than_420/']

def file_max_num():
    max_num = 0
    for dir in dirs:
        range_dir = os.path.join(file_dir, dir)
        files = os.listdir(range_dir)
        num = len(files)
        # print(num)
        if num > max_num:
            max_num = num
    return max_num

def extend():
    # 晶胞结构旋转只有23种情况
    a = np.array([[1, -3, 2], [2, -3, -1], [-3, -2, -1], [-1, -2, 3], [3, -2, 1], [-2, -3, 1], [-3, 2, 1], [1, 3, -2],
                  [2, -1, 3], [3, -1, -2], [-1, -3, -2], [-3, -1, 2], [-3, 1, -2], [-2, 1, 3], [3, 2, -1], [3, 1, 2],
                  [2, 1, -3], [1, -2, -3], [-2, -1, -3], [-1, 3, 2], [2, 3, 1], [-2, 3, -1], [-1, 2, -3]])
    max_num = file_max_num()    # 获取不同吸附区间的最大文件数，以此为基准拓展数据
    for dir in dirs:    # 吸附值不同的文件目录
        print(dir)
        range_dir = os.path.join(file_dir, dir)
        num = 0
        sub_dirs = os.listdir(range_dir)
        for sub_dir in sub_dirs:    # 每一个cif文件
            index_random = []  # 计数
            ori_sub_dir = sub_dir
            for abc in range(int(max_num / len(sub_dirs)) - 1):  # 拓展次数
                x_index = 0
                y_index = 0
                z_index = 0
                random_number = random.randint(0, 22)
                while random_number in index_random:    # 在旋转晶胞的时候要不同。如果已经旋转过，选择另一种
                    random_number = random.randint(0, 22)
                index_random.append(random_number)
                new_cif_name = 'expand' + str(abc) + str(ori_sub_dir)
                # print(new_cif_name)
                if len(index_random) == 24:
                    print('extand num outside!')
                    break
                whirl_matrix = a[random_number]
                if 1 in whirl_matrix:
                    x_place = int(np.argwhere(whirl_matrix == 1))   # 返回数组里1的索引(下标)
                    x_index = 1
                else:
                    x_place = int(np.argwhere(whirl_matrix == -1))
                    x_index = -1
                if 2 in whirl_matrix:
                    y_place = int(np.argwhere(whirl_matrix == 2))
                    y_index = 1
                else:
                    y_place = int(np.argwhere(whirl_matrix == -2))
                    y_index = -1
                if 3 in whirl_matrix:
                    z_place = int(np.argwhere(whirl_matrix == 3))
                    z_index = 1
                else:
                    z_place = int(np.argwhere(whirl_matrix == -3))
                    z_index = -1
                new_cif_name = os.path.join(range_dir, new_cif_name)
                new_cif = open(new_cif_name, 'w')
                sub_dir = os.path.join(range_dir, ori_sub_dir)
                if os.path.isfile(sub_dir):
                    file = open(sub_dir)
                for i in range(0, 8):
                    line = file.readline()
                    new_cif.write(line)
                #  length_a
                line = file.readline()
                line_vec = line.strip("\n").split("\t")
                length_a = float(line_vec[1])
                # length_b
                line = file.readline()
                line_vec = line.strip("\n").split("\t")
                length_b = float(line_vec[1])
                # length_c
                line = file.readline()
                line_vec = line.strip("\n").split("\t")
                length_c = float(line_vec[1])
                # alpha
                line = file.readline()
                line_vec = line.strip("\n").split("\t")
                angle_alpha = float(line_vec[1])
                # beta
                line = file.readline()
                line_vec = line.strip("\n").split("\t")
                angle_beta = float(line_vec[1])
                # gamma
                line = file.readline()
                line_vec = line.strip("\n").split("\t")
                angle_gamma = float(line_vec[1])
                if whirl_matrix[0] == 1 or whirl_matrix[0] == -1:
                    cif_line = '_cell_length_a' + '\t' + str(length_a)
                    new_cif.write(cif_line + '\n')
                    if whirl_matrix[1] == 2 or whirl_matrix[1] == -2:
                        cif_line = '_cell_length_b' + '\t' + str(length_b)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_length_c' + '\t' + str(length_c)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_alpha' + '\t' + str(angle_alpha)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_beta' + '\t' + str(angle_beta)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_gamma' + '\t' + str(angle_gamma)
                        new_cif.write(cif_line + '\n')
                    else:
                        cif_line = '_cell_length_b' + '\t' + str(length_c)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_length_c' + '\t' + str(length_b)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_alpha' + '\t' + str(angle_alpha)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_beta' + '\t' + str(angle_gamma)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_gamma' + '\t' + str(angle_beta)
                        new_cif.write(cif_line + '\n')
                elif whirl_matrix[0] == 2 or whirl_matrix[0] == -2:
                    cif_line = '_cell_length_a' + '\t' + str(length_b)
                    new_cif.write(cif_line + '\n')
                    if whirl_matrix[1] == 1 or whirl_matrix[1] == -1:
                        cif_line = '_cell_length_b' + '\t' + str(length_a)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_length_c' + '\t' + str(length_c)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_alpha' + '\t' + str(angle_beta)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_beta' + '\t' + str(angle_alpha)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_gamma' + '\t' + str(angle_gamma)
                        new_cif.write(cif_line + '\n')
                    else:
                        cif_line = '_cell_length_b' + '\t' + str(length_c)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_length_c' + '\t' + str(length_a)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_alpha' + '\t' + str(angle_beta)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_beta' + '\t' + str(angle_gamma)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_gamma' + '\t' + str(angle_alpha)
                        new_cif.write(cif_line + '\n')
                else:
                    cif_line = '_cell_length_a' + '\t' + str(length_c)
                    new_cif.write(cif_line + '\n')
                    if whirl_matrix[1] == 1 or whirl_matrix[1] == -1:
                        cif_line = '_cell_length_b' + '\t' + str(length_a)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_length_c' + '\t' + str(length_b)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_alpha' + '\t' + str(angle_gamma)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_beta' + '\t' + str(angle_alpha)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_gamma' + '\t' + str(angle_beta)
                        new_cif.write(cif_line + '\n')
                    else:
                        cif_line = '_cell_length_b' + '\t' + str(length_b)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_length_c' + '\t' + str(length_a)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_alpha' + '\t' + str(angle_gamma)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_beta' + '\t' + str(angle_beta)
                        new_cif.write(cif_line + '\n')
                        cif_line = '_cell_angle_gamma' + '\t' + str(angle_alpha)
                        new_cif.write(cif_line + '\n')
                for i in range(14, 20):
                    line = file.readline()
                    new_cif.write(line)
                line = file.readline()
                while line:
                    line_vec = line.strip("\n").split("\t")
                    abc_point = np.array([float(line_vec[2]), float(line_vec[3]), float(line_vec[4])])  # 原子 x y z
                    while abc_point[0] > 1 or abc_point[0] < 0:
                        if abc_point[0] > 1:
                            abc_point[0] -= 1
                        else:
                            abc_point[0] += 1
                    while abc_point[1] > 1 or abc_point[1] < 0:
                        if abc_point[1] > 1:
                            abc_point[1] -= 1
                        else:
                            abc_point[1] += 1
                    while abc_point[2] > 1 or abc_point[2] < 0:
                        if abc_point[2] > 1:
                            abc_point[2] -= 1
                        else:
                            abc_point[2] += 1
                    new_point = np.array([1, 1, 1], dtype=float)
                    new_point[x_place] = x_index * abc_point[0]
                    new_point[y_place] = y_index * abc_point[1]
                    new_point[z_place] = z_index * abc_point[2]
                    cif_line = str(line_vec[0]) + '\t' + str(line_vec[1]) + '\t' + str(
                        round(float(new_point[0]), 6)) + '\t' + str(
                        round(float(new_point[1]), 6)) + '\t' + str(
                        round(float(new_point[2]), 6))
                    new_cif.write(cif_line + '\n')
                    line = file.readline()
                new_cif.close()
                num += 1
                file.close()

def main():
    extend()

if __name__ == '__main__':
    main()
