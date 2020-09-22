import os
import pandas as pd

# filedir = 'E:/project/mofs-data/xyz_data/test_data'
# filedir = 'E:/project/mofs-data/xyz_data/train_data'
# gene_uptake_path = "E:/project/mofs-data/gene_uptake.xlsx"
# filedir = 'E:/project/mofs-data/test'

filedir = '/data/user10110/mofs-data/later_data/train_data'
xyzdir = '/data/user10110/mofs-data/xyz_data/train_data'
gene_uptake_path = "/data/user10110/mofs-data/gene_uptake.xlsx"

dirs = ['data_less_than_60/', 'data_60_to_120/', 'data_120_to_180/', 'data_180_to_240/', 'data_240_to_300/',
        'data_300_to_360/', 'data_360_to_420/', 'data_more_than_420/']

multiple = 30
class Atomstation(object):
    def __init__(self, name, x, y, z):
        self.name = name
        self.x = round(float(x) * multiple + 5.340426 * multiple)   # 向右平移加扩大multiple倍,正规化
        self.y = round(float(y) * multiple + 4.917102 * multiple)
        self.z = round(float(z) * multiple + 4.720307 * multiple)
atoms = {}  # {'Cu', 'N', 'Cl', 'F', 'C', 'V', 'Br', 'Zr', 'O', 'H', 'Zn'}
atoms["Cu"] = 29
atoms["N"] = 7
atoms["Cl"] = 17
atoms["F"] = 9
atoms["C"] = 6
atoms["V"] = 23
atoms["Br"] = 35
atoms["Zr"] = 40
atoms["O"] = 8
atoms["H"] = 1
atoms["Zn"] = 30

def main():
    xydir = os.path.join(xyzdir, 'x,yflat')
    yzdir = os.path.join(xyzdir, 'y,zflat')
    xzdir = os.path.join(xyzdir, 'x,zflat')
    if not os.path.isdir(xydir):
        os.makedirs(xydir)
    if not os.path.isdir(yzdir):
        os.makedirs(yzdir)
    if not os.path.isdir(xzdir):
        os.makedirs(xzdir)

    num = 0
    for dir in dirs:    # 每一个吸附区间目录文件
        print(dir)
        range_dir = os.path.join(filedir, dir)
        cif_dirs = os.listdir(range_dir)
        for cif_dir in cif_dirs:    # 每一个cif文件名称
            num += 1
            struct = cif_dir.split(".")[0]
            struct = struct[struct.find("_", struct.find("_") + 1) + 1:]
            struct_vals = pd.DataFrame(pd.read_excel(gene_uptake_path))  # 读取xlsx文件中原子结构的吸附值
            val = struct_vals.query("gene_structure == '" + struct + "'")["uptake(CH4)"].astype(float)  # 吸附值
            filename = struct + str(val.values) + str(11 * multiple) + 'x' + str(11 * multiple) + '.txt'    # 11个原子
            cif_path = os.path.join(range_dir, cif_dir)
            if os.path.isfile(cif_path):
                file = open(cif_path)
                filexy = open(os.path.join(xydir, (str(num) + 'xy' + filename)), 'a')
                fileyz = open(os.path.join(yzdir, (str(num) + 'yz' + filename)), 'a')
                filexz = open(os.path.join(xzdir, (str(num) + 'xz' + filename)), 'a')

                for i in range(0, 20):
                    line = file.readline()
                line = file.readline()
                while line:
                    line_vec = line.strip("\n").split("\t")
                    a = Atomstation(line_vec[1], line_vec[2], line_vec[3], line_vec[4])
                    filexy.write(str(a.x) + ',' + str(a.y) + ',' + str(atoms[a.name]))
                    filexy.write("\n")
                    fileyz.write(str(a.y) + ',' + str(a.z) + ',' + str(atoms[a.name]))
                    fileyz.write("\n")
                    filexz.write(str(a.x) + ',' + str(a.z) + ',' + str(atoms[a.name]))
                    filexz.write("\n")
                    line = file.readline()
                filexy.close()
                fileyz.close()
                filexz.close()
    print(num)

if  __name__ == '__main__':
    main()

