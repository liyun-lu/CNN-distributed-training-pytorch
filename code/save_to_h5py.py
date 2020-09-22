import os
import torch
import numpy as np
from scipy.sparse import lil_matrix
import h5py

data_path = '/data/user10110/mofs-data/xyz_data/test_data'
h5py_path = '/data/user10110/mofs-data/test_h5py.h5'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(nums, file_path):
    tensora = []
    absorb = []
    multiple = 330
    dataxy_path = os.path.join(file_path, 'x,yflat')
    datayz_path = os.path.join(file_path, 'y,zflat')
    dataxz_path = os.path.join(file_path, 'x,zflat')
    for num in nums:
        i = 1
        k = 1   # 用来记录加载到第几个文件
        sub_dirs = os.listdir(dataxy_path)
        for sub_dir in sub_dirs:
            if k < num:
                k += 1
            else:
                if i < 2:
                    sub_vec = sub_dir.split("xy")[1]
                    num_name = sub_dir.split('xy')[0]
                    datayz_name = num_name + 'yz' + sub_vec
                    dataxz_name = num_name + 'xz' + sub_vec
                    sub_vec = sub_vec.split('[')[1].split(']')  # 吸附值
                    absorb.append(sub_vec[0])

                    matrixyz = lil_matrix((multiple, multiple), dtype=np.uint16)
                    sub_dir = os.path.join(dataxy_path, sub_dir)
                    filexy = open(sub_dir, 'r')
                    line = filexy.readline()
                    while line:
                        line_vec = line.strip("\n").split(",")
                        matrixyz[int(line_vec[0]) - 1, int(line_vec[1]) - 1] = int(line_vec[2]) + matrixyz[
                            int(line_vec[0]) - 1, int(line_vec[1]) - 1]
                        line = filexy.readline()
                    filexy.close()
                    tensora.append(matrixyz)

                    matrixyz = lil_matrix((multiple, multiple), dtype=np.uint16)
                    sub_dir = os.path.join(datayz_path, datayz_name)
                    fileyz = open(sub_dir, 'r')
                    line = fileyz.readline()
                    while line:
                        line_vec = line.strip("\n").split(",")
                        matrixyz[int(line_vec[0]) - 1, int(line_vec[1]) - 1] = int(line_vec[2]) + matrixyz[
                            int(line_vec[0]) - 1, int(line_vec[1]) - 1]
                        line = fileyz.readline()
                    fileyz.close()
                    tensora.append(matrixyz)

                    matrixyz = lil_matrix((multiple, multiple), dtype=np.uint16)
                    sub_dir = os.path.join(dataxz_path, dataxz_name)
                    filexz = open(sub_dir, 'r')
                    line = filexz.readline()
                    while line:
                        line_vec = line.strip("\n").split(",")
                        matrixyz[int(line_vec[0]) - 1, int(line_vec[1]) - 1] = int(line_vec[2]) + matrixyz[
                            int(line_vec[0]) - 1, int(line_vec[1]) - 1]
                        line = filexz.readline()
                    filexz.close()
                    tensora.append(matrixyz)
                    i = i + 1
                else:
                    break

    x_batch = np.empty((3, multiple, multiple), dtype=np.uint16)  # 构建一个数据，三个通道，行、列
    y_batch = np.empty((1,))
    count = 0
    for k in range(0, 3):
        tensora[count] = tensora[count].toarray()
        tensora[count] = tensora[count].reshape(1, multiple, multiple)
        x_batch[k:, :, :] = tensora[count]
        count = count + 1
    y_batch[:, ] = float(absorb[int(count / 3) - 1])
    # print(y_batch)
    return (x_batch, y_batch)

def save_to_h5py():
    f = h5py.File(h5py_path, 'w')
    img_list = []
    lable_list = []
    for i in range(14161):  # 56652 14161
        image, lable = load_data([i + 1], data_path)
        img_list.append(image)
        lable_list.append(lable)
        # print(data, lable)
    f.create_dataset('inputs', data=img_list, compression='gzip', compression_opts=9)   # 数据压缩和压缩级别
    f.create_dataset('lables', data=lable_list, compression='gzip', compression_opts=9)
    f.close()

def main():
    print('start')
    save_to_h5py()
    print('have save to h5py')

if __name__ == '__main__':
    main()





