# -*-coding:utf-8-*-
# 分类程序(for LIGO DATA)v1.0，杨楠，2018、10、05
# 训练神经网络
# 建议使用pycharm运行本程序。
"""
使用方法：
此脚本为训练神经网络，直接运行就可以了。
"""
from data_proc import DataProc
from keras.models import load_model

# 加载神经网络模型
model = load_model('cnn_for_LIGO_0.h5')
# 创建数据处理类
dp = DataProc(model=model)  # 或者创建对象dp后加载model： dp.load_network_model('cnn_for_LIGO_0.h5')
dp.load_norm_train_data(num_files=2000)  # 加载归一化的训练数据
dp.load_norm_test_data(num_files=100)  # 加载归一化的测试数据

# 数据reshape，准备进行训练
dp.train_data_reshape()
dp.test_data_reshape()

# 训练神经网络，batch要设置的与网络一致，epochs是重复次数。
dp.train_network(batch_size=1000, epochs=200)
