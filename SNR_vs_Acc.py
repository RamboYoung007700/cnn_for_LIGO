# -*-coding:utf-8-*-
# 分类程序(for LIGO DATA)v1.0，杨楠，2018、10、05
# 信噪比与准确率
# 建议使用pycharm运行本程序。
"""
使用方法：
此脚本用来计算信噪比与准确率的关系。直接运行本脚本就可以。
使用技巧：
1.如果之前已经进行过测试集的预处理，这里不需要重复处理。
2.计算好snr列表后，就可以注释掉这部分代码，因为snr不会变了。
"""

from data_proc import DataProc
from keras.models import load_model
import numpy as np

model = load_model('cnn1004.h5')  # 加载训练好的神经网络
dp = DataProc(model=model)  # 或者创建对象dp后加载model： dp.load_network_model('cnn_for_LIGO_0.h5')

# 下面为测试数据集合预处理，如果之前在data_pretreatment已经处理过，则这里不需要再进行预处理。
dp.load_mu_sigma()  # 加载归一化参数
dp.read_original_test_data(num_files=1000, buffer=10)  # 读取原始测试集数据
dp.data_flatten('data/workdatatest', 10)  # 拉直数据
dp.load_flatten_test_data(num_files=10)  # 加载拉直的数据
dp.label_test_one_hot()  # 标签hot编码
dp.test_feature_normalize()  # 归一化测试数据
dp.save_test_norm_data(buffer=10)  # 保存归一化后的数据

dp.get_files_SNR(num_files=100)  # 得到SNR列表并保存

# 下面利用训练好的模型，对测试集进行预测。
dp.load_norm_test_data(num_files=200)  # 加载归一化后的测试集
dp.test_data_reshape()  # 维度重整
dp.get_acc_list()  # 得到准确率列表，并保存

# 到这里，snr列表与acc列表都计算完成并保存在文件中了。
############################################分割线############################################

# 下面进行snr与acc关系的计算
snr = np.loadtxt('snr_list.txt', dtype=np.float64)  # 加载snr列表
acc = np.loadtxt('acc_list.txt', dtype=np.float64)  # 加载acc列表
m = len(snr)  # 样本总数


# 定义结果类
class MyRsultData():
    def __init__(self, snr, acc):
        self.snr = snr
        self.acc = acc


# 建立结果列表，列表中每个元素是一个样本的结果对象，结果对象有两个属性，一个是snr，一个是acc
result_list = []

# 下面这个循环把结果列表的元素都创建出来。
for i in range(m):
    temp = MyRsultData(snr[i], acc[i])
    result_list.append(temp)

# 按照snr从小到大排列结果列表的元素
result_list.sort(key=lambda x: x.snr)

# 利用列表表达式，创建不同snr区间的结果元素子列表
snr_15 = [x for x in result_list if 0 < x.snr <= 15]
snr_20 = [x for x in result_list if 15 < x.snr <= 20]
snr_25 = [x for x in result_list if 20 < x.snr <= 25]
snr_30 = [x for x in result_list if 25 < x.snr <= 30]
snr_35 = [x for x in result_list if 30 < x.snr <= 35]
snr_40 = [x for x in result_list if 35 < x.snr <= 40]
snr_60 = [x for x in result_list if 40 < x.snr <= 60]
snr_0 = [x for x in result_list if x.snr == 0]


# 计算一个子列表的准确率
def get_acc(rslt_list):
    counter = 0
    for rslt in rslt_list:
        if rslt.acc == 1:
            counter += 1
    lenth = len(rslt_list)
    if lenth == 0:
        lenth = 0.0001
    return round(counter / lenth, 2), lenth


# 为输出做格式准备
index = ['0-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-60', '0(纯噪音)']
rslt = [get_acc(snr_15), get_acc(snr_20), get_acc(snr_25), get_acc(snr_30), get_acc(snr_35), get_acc(snr_40),
        get_acc(snr_60), get_acc(snr_0)]

# 输出结果
for i in range(len(index)):
    print('snr: ' + str(index[i]) + '  模型预测准确率：' + str(rslt[i][0]) + '  此分组总样本数:' + str(rslt[i][1]))
