# -*-coding:utf-8-*-
# 分类程序(for LIGO DATA)v1.0，杨楠，2018、10、05
# 数据预处理
# 建议使用pycharm运行本程序。
"""
使用方法：
此脚本为数据预处理，讲训练数据放在/data/1下，测试数据放在/data/tdata/1下，
然后直接运行就可以了。
使用技巧： 可以先生成训练集，然后把所有有关测试集的代码注释掉。在运行训练集的预处理程序时，
           用另一台电脑运行模拟程序，来生成测试集数据。
注意：文件数量参数要根据自己需要来设定。我这里的参数是针对20000个训练数据文件，1000个测试数据文件.
"""
# 导入数据处理类
from data_proc import DataProc

# 根据类创建数据处理对象
dp = DataProc()
dp.read_original_train_data(num_files=20000, buffer=10)  # 读原始训练数据 （simulation那个程序生成的数据）
dp.read_original_test_data(num_files=1000, buffer=10)  # 读原始测试数据

# 首尾相接，拉直训练集.第二个参数是文件数，如果不知道如何计算，可以先运行上面的程序，
# 然后查看文件夹下有多少个文件，再填入第二个参数。文件数计算公式：num_files/buffer
dp.data_flatten('data/workdata', 2000)
dp.data_flatten('data/workdatatest', 100)  # 首位详解，拉直测试集

# 一个执行技巧是，先把下面的代码注释，执行上面的，执行完上面的就注释上面的代码，然后执行下面的。
dp.load_flatten_train_data(num_files=2000)  # 加载拉直后的训练集
dp.load_flatten_test_data(num_files=100)  # 加载拉直后的测试集

# 标签改one hot格式
dp.label_train_one_hot()
dp.label_test_one_hot()
# z归一化
dp.train_feature_normalize()
dp.test_feature_normalize()
# 存储归一化后的数据
dp.save_train_norm_data(buffer=10)
dp.save_test_norm_data(buffer=10)
# 存储归一化参数
dp.save_mu_sigma()
