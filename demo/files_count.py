import random
import os
# filedir = "E:/project/mofs-data/test"
# dirs = ['data_less_than_40/', 'data_360_to_400/', 'data_400_to_440/']
filedir = "E:/project/mofs-data/xyz_data/test_data/x,yflat"
# filedir = "E:/project/mofs-data/later_data/test_data"

file_max_num = 0
total = 0
files = os.listdir(filedir)
num = len(files)

print(num)

# print(file_max_num)