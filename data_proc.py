# -*-coding:utf-8-*-
# 分类程序(for LIGO DATA)v1.0，杨楠，2018、10、05
# 建议使用pycharm运行本程序。
"""
使用方法：
此脚本为数据处理类脚本，其他脚本导入本脚本后可以创建数据处理对象。
"""
import numpy as np
import os
from keras.models import load_model
from scipy import signal
import matplotlib.mlab as mlab


class DataProc():

    def __init__(self, data_features=4096, sample_frequency=2048, model=None):
        self.data_features = data_features
        self.sample_frequency = sample_frequency
        self.model = model
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.mu = None
        self.sigma = None
        self.x_train_norm = None
        self.x_test_norm = None
        self.y_train_hot = None
        self.y_test_hot = None
        self.x_test_snr = None
        self.snr_list = []
        self.acc_list = []
        self.acc = None

    def load_network_model(self, path):
        self.model = load_model(path)

    def mkdir(self, path):
        workpath = os.getcwd()
        path = workpath + '/' + path
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path + ' 自动创建成功！将进行读写操作！')
            return True
        else:
            print(path + ' 目录已存在，直接进行读写操作！')
            return False

    def read_text_file(self, file_name):
        """本函数读取一个文件"""
        data = np.loadtxt(file_name, dtype=np.float64)
        return data

    def read_original_train_data(self, file_folder='data', num_folders=1, num_files=100, buffer=10,
                                 snr=False, save_path='data/workdata/'):
        """本函数读取原始数据，连接后按buffer保存到文件中"""
        counter = 0
        data = []
        for j in range(num_folders):
            for i in range(num_files):
                file_name = file_folder + '/' + str(j + 1) + '/' + str(i) + '_data.txt'
                try:
                    temp = self.read_text_file(file_name)
                except IOError:
                    temp = data[-1]

                temp = np.mat(temp)
                data.append(temp)
                print('读入第' + str(j + 1) + '部分数据文件: ' + str(i) + '/' + str(num_files - 1))
                counter += 1
                if counter >= buffer:
                    self.save_X_y(j, i, buffer, num_files, data, snr, save_path)
                    data = []
                    counter = 0

    def save_X_y(self, foldern, filen, buffer, num_files_from, data, snr=False, save_path='data/workdata/'):
        """从数据中得到X，y，并存储"""
        num_files = len(data)
        n = len(data[0])
        X = np.zeros((1, n))
        y = []
        if snr:
            X_snr = np.zeros((1, n))
        for i in range(num_files):
            temp = data[i][:, 1]
            X = np.r_[X, temp.T]
            temp = data[i][:, 3]
            X = np.r_[X, temp.T]
            temp = data[i][:, 5]
            X = np.r_[X, temp.T]
            y.append(1)

            temp = data[i][:, 1] - data[i][:, 2]
            X = np.r_[X, temp.T]
            temp = data[i][:, 3] - data[i][:, 4]
            X = np.r_[X, temp.T]
            temp = data[i][:, 5] - data[i][:, 6]
            X = np.r_[X, temp.T]
            y.append(2)

            if snr:
                temp = data[i][:, 0]
                X_snr = np.r_[X_snr, temp.T]
                temp = data[i][:, 1]
                X_snr = np.r_[X_snr, temp.T]
                temp = data[i][:, 2]
                X_snr = np.r_[X_snr, temp.T]
                temp = data[i][:, 3]
                X_snr = np.r_[X_snr, temp.T]
                temp = data[i][:, 4]
                X_snr = np.r_[X_snr, temp.T]
                temp = data[i][:, 5]
                X_snr = np.r_[X_snr, temp.T]
                temp = data[i][:, 6]
                X_snr = np.r_[X_snr, temp.T]

            print("已经转化数据： " + str(i + 1) + '/' + str(num_files))
        X = np.delete(X, 0, 0)
        y = np.matrix(y).T
        if snr:
            X_snr = np.delete(X_snr, 0, 0)
        print('保存数据...')
        path = save_path
        self.mkdir(path)
        name = int((filen + 1) / buffer + foldern * (num_files_from / buffer))
        np.savetxt(path + str(name) + 'X.txt', X)
        np.savetxt(path + str(name) + 'y.txt', y)
        if snr:
            path += 'snr/'
            self.mkdir(path)
            np.savetxt(path + str(name) + 'X.txt', X_snr)
        print('保存完成')

    def read_original_test_data(self, file_folder='data/tdata', num_folders=1, num_files=100, buffer=10,
                                snr=True, save_path='data/workdatatest/'):
        """本函数读取原始测试集数据，连接后按buffer保存到文件中"""
        self.read_original_train_data(file_folder, num_folders, num_files, buffer, snr, save_path)

    def data_flatten(self, file_folder, n_file):
        '''数据首位相接，展平'''
        X = np.loadtxt(file_folder + '/1X.txt', dtype=np.float64)
        # X = np.loadtxt(file_folder + '/1X.txt', dtype=np.float64)
        X = np.mat(X)
        m, n = X.shape
        n = int(n * 3)
        m = int(m / 3)
        for i in range(n_file):
            X = np.loadtxt(file_folder + '/' + str(i + 1) + 'X.txt', dtype=np.float64)
            X = np.mat(X)
            X = np.reshape(X, (m, n))
            np.savetxt(file_folder + '/' + str(i + 1) + 'X.txt', X)
            print("已转化完" + str(i + 1) + '/' + str(n_file))

    def load_flatten_train_data(self, file_folder='data/workdata', num_files=10):
        '''读取首位相接的训练数据'''
        X = np.loadtxt(file_folder + '/1X.txt', dtype=np.float64)
        X = np.mat(X)
        m, n = X.shape
        data_X = np.zeros((1, n))
        data_y = np.zeros((1, 1))
        print('开始读入数据,较为缓慢,请耐心等待...')
        for i in range(num_files):
            X = np.loadtxt(file_folder + '/' + str(i + 1) + 'X.txt', dtype=np.float64)
            y = np.loadtxt(file_folder + '/' + str(i + 1) + 'y.txt', dtype=np.float64)
            X = np.mat(X)
            y = np.mat(y).T
            data_X = np.vstack((data_X, X))
            data_y = np.vstack((data_y, y))
            print("已经读完" + str(i + 1) + '/' + str(num_files))
        data_X = np.delete(data_X, 0, 0)
        data_y = np.delete(data_y, 0, 0)
        print('读取完毕!')
        self.x_train = data_X
        self.y_train = data_y

    def load_flatten_test_data(self, file_folder='data/workdatatest', num_files=10):
        '''读取首位相接的测试数据'''
        X = np.loadtxt(file_folder + '/1X.txt', dtype=np.float64)
        X = np.mat(X)
        m, n = X.shape
        data_X = np.zeros((1, n))
        data_y = np.zeros((1, 1))
        print('开始读入数据,较为缓慢,请耐心等待...')
        for i in range(num_files):
            X = np.loadtxt(file_folder + '/' + str(i + 1) + 'X.txt', dtype=np.float64)
            y = np.loadtxt(file_folder + '/' + str(i + 1) + 'y.txt', dtype=np.float64)
            X = np.mat(X)
            y = np.mat(y).T
            data_X = np.vstack((data_X, X))
            data_y = np.vstack((data_y, y))
            print("已经读完" + str(i + 1) + '/' + str(num_files))
        data_X = np.delete(data_X, 0, 0)
        data_y = np.delete(data_y, 0, 0)
        print('读取完毕!')
        self.x_test = data_X
        self.y_test = data_y

    def label_train_one_hot(self, num_labels=2):
        """本函数将标签列向量转化为一个矩阵，方便神经网络计算"""
        y = self.y_train
        m = np.shape(y)[0]
        y_result = np.zeros((num_labels, 1))
        for i in range(m):
            y_col = np.zeros((num_labels, 1))
            y_col[int(y[i, 0] - 1), 0] = 1  # 标签10则第9行为1，标签为1则第0行为1.
            y_result = np.c_[y_result, y_col]
        y_result = np.delete(y_result, 0, axis=1)
        self.y_train_hot = y_result.T

    def label_test_one_hot(self, num_labels=2):
        """本函数将标签列向量转化为一个矩阵，方便神经网络计算"""
        y = self.y_test
        m = np.shape(y)[0]
        y_result = np.zeros((num_labels, 1))
        for i in range(m):
            y_col = np.zeros((num_labels, 1))
            y_col[int(y[i, 0] - 1), 0] = 1  # 标签10则第9行为1，标签为1则第0行为1.
            y_result = np.c_[y_result, y_col]
        y_result = np.delete(y_result, 0, axis=1)
        self.y_test_hot = y_result.T

    def train_feature_normalize(self):
        """z归一化。每一列都减去平均值除以标准差。"""
        X = self.x_train
        m, n = X.shape
        mu = np.mean(X, 0)
        sigma = np.std(X, axis=0, ddof=1)  # 这里有坑，记得加ddof参数
        r, c = sigma.shape
        for i in range(c):
            if sigma[0, i] == 0:
                sigma[0, i] = 0.000000000000000001 ** 4
        for i in range(n):
            X[:, i] = (X[:, i] - mu[0, i]) / sigma[0, i]
            print('已经归一化数据：' + str(i + 1) + '/' + str(n))
        self.x_train_norm = X
        self.mu = mu
        self.sigma = sigma

    def test_feature_normalize(self):
        """z归一化。每一列都减去平均值除以标准差。"""
        X = self.x_test
        m, n = X.shape
        mu = self.mu
        sigma = self.sigma

        for i in range(n):
            X[:, i] = (X[:, i] - mu[0, i]) / sigma[0, i]
            print('已经归一化数据：' + str(i + 1) + '/' + str(n))
        self.x_test_norm = X

    def save_train_norm_data(self, path='data/normdata/', buffer=10):
        self.mkdir(path)
        X = self.x_train_norm
        y = self.y_train_hot
        m, n = X.shape
        num_files = int(m / buffer)
        for i in range(num_files):
            temp_X = X[buffer * i:buffer * (i + 1), :]
            temp_y = y[buffer * i:buffer * (i + 1), :]
            np.savetxt(path + str(i + 1) + 'X.txt', temp_X)
            np.savetxt(path + str(i + 1) + 'y.txt', temp_y)
            print('normalization data saved: ' + str(i + 1) + '/' + str(num_files))

    def save_test_norm_data(self, path='data/normdatatest/', buffer=10):
        self.mkdir(path)
        X = self.x_test_norm
        y = self.y_test_hot
        m, n = X.shape
        num_files = int(m / buffer)
        for i in range(num_files):
            temp_X = X[buffer * i:buffer * (i + 1), :]
            temp_y = y[buffer * i:buffer * (i + 1), :]
            np.savetxt(path + str(i + 1) + 'X.txt', temp_X)
            np.savetxt(path + str(i + 1) + 'y.txt', temp_y)
            print('normalization data saved: ' + str(i + 1) + '/' + str(num_files))

    def save_mu_sigma(self):
        np.savetxt('mu.txt', self.mu)
        np.savetxt('sigma.txt', self.sigma)

    def load_mu_sigma(self):
        self.mu = np.mat(np.loadtxt('mu.txt', dtype=np.float64))
        self.sigma = np.mat(np.loadtxt('sigma.txt', dtype=np.float64))

    def load_norm_train_data(self, file_folder='data/normdata', num_start=1, num_files=10):
        """本函数读取归一化数据文件夹"""
        num_features = int(self.data_features * 3)
        data_X = np.zeros((1, num_features))
        data_y = np.zeros((1, 2))

        for i in range(num_start, num_start + num_files):
            X = np.loadtxt(file_folder + '/' + str(i) + 'X.txt', dtype=np.float64)
            y = np.loadtxt(file_folder + '/' + str(i) + 'y.txt', dtype=np.float64)
            X = np.mat(X)
            y = np.mat(y)

            data_X = np.vstack((data_X, X))
            data_y = np.vstack((data_y, y))
            print("已经读完" + str(i) + '/' + str(num_start + num_files - 1))
        data_X = np.delete(data_X, 0, 0)
        data_y = np.delete(data_y, 0, 0)
        self.x_train = data_X
        self.y_train = data_y

    def load_norm_test_data(self, file_folder='data/normdatatest', num_start=1, num_files=10):
        """本函数读取归一化数据文件夹"""
        num_features = int(self.data_features * 3)
        data_X = np.zeros((1, num_features))
        data_y = np.zeros((1, 2))

        for i in range(num_start, num_start + num_files):
            X = np.loadtxt(file_folder + '/' + str(i) + 'X.txt', dtype=np.float64)
            y = np.loadtxt(file_folder + '/' + str(i) + 'y.txt', dtype=np.float64)
            X = np.mat(X)
            y = np.mat(y)

            data_X = np.vstack((data_X, X))
            data_y = np.vstack((data_y, y))
            print("已经读完" + str(i) + '/' + str(num_start + num_files - 1))
        data_X = np.delete(data_X, 0, 0)
        data_y = np.delete(data_y, 0, 0)
        self.x_test = data_X
        self.y_test = data_y

    def train_data_reshape(self):
        '''打断首尾相接，生成4维数组，方便训练神经网络'''
        m, n = self.x_train.shape
        x = np.zeros((m, 3, self.data_features, 1))
        for i in range(m):
            for j in range(3):
                for k in range(self.data_features):
                    x[i, j, k, 0] = self.x_train[i, int(j * 3 + k)]
            print('已经reshape数据：' + str(i + 1) + '/' + str(m))
        self.x_train = x

    def test_data_reshape(self):
        '''打断首尾相接，生成4维数组，方便神经网络进行预测'''
        m, n = self.x_test.shape
        x = np.zeros((m, 3, self.data_features, 1))
        for i in range(m):
            for j in range(3):
                for k in range(self.data_features):
                    x[i, j, k, 0] = self.x_test[i, int(j * 3 + k)]
            print('已经reshape数据：' + str(i + 1) + '/' + str(m))
        self.x_test = x

    def train_network(self, batch_size=10, epochs=200, save_path='data/model', save_name='savemodel.h5'):
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)
        self.mkdir(save_path)
        path = save_path + '/' + save_name
        self.model.save(path)
        print('训练完成，训练后的模型保存在：' + path + '文件中')
        self.get_accuracy(self.model.predict(self.x_test), self.y_test)
        print('准确率为:' + str(self.acc))
        np.savetxt('acc_list.txt', self.acc_list)

    def get_acc_list(self):
        self.get_accuracy(self.model.predict(self.x_test), self.y_test)
        print('准确率为:' + str(self.acc))
        np.savetxt('acc_list.txt', self.acc_list)
        print('已经保存acc_list在工作路径下的acc_list.txt文件')

    def get_accuracy(self, rslt, y):
        acc_list = []
        m, n = rslt.shape
        m1, n1 = y.shape
        if m != m1 or n != n1:
            print("标签矩阵与结果矩阵维度不匹配")
        counter = 0
        rslt = self.my_softmax(rslt)
        for i in range(m):
            if (rslt[i, :] == y[i, :]).all():
                counter += 1
                acc_list.append(1)
            else:
                acc_list.append(0)
        self.acc = counter / m
        self.acc_list = acc_list

    def my_softmax(self, y):
        m, n = y.shape
        rslt = np.zeros((m, n))
        for i in range(m):
            col = np.where(y[i, :] == np.max(y[i, :]))
            rslt[i, col[0][0]] = 1
        return rslt

    def get_files_SNR(self, file_folder='data/workdatatest/snr', num_files=100, zeros_fill=True):
        snr_list = []
        for i in range(num_files):
            filepath = file_folder + '/' + str(i + 1) + 'X.txt'
            sl = self.get_file_SNR(filepath, self.sample_frequency, zeros_fill)
            snr_list += sl
            print('已经计算完成：' + str(i + 1) + '/' + str(num_files))
        self.snr_list = np.mat(snr_list).T
        np.savetxt('snr_list.txt', self.snr_list)
        print('已经保存snr列表文件在：snr_list.txt')

    def load_snr_list(self):
        self.snr_list = np.mat(np.loadtxt('snr_list.txt', dtype=np.float64))

    def get_file_SNR(self, file_path, fs, zeros_fill=False):
        '''计算一个文件的snr，zeros fill是填充纯噪音的snr为0'''
        data = np.loadtxt(file_path)
        m, n = data.shape
        snr_list = []
        for i in range(m // 7):
            t = data[i * 7 + 0, :]
            hn = data[i * 7 + 1, :]
            ht = data[i * 7 + 2, :]
            snr1 = self.get_SNR(t, hn, ht, fs)

            hn = data[i * 7 + 3, :]
            ht = data[i * 7 + 4, :]
            snr2 = self.get_SNR(t, hn, ht, fs)

            hn = data[i * 7 + 5, :]
            ht = data[i * 7 + 6, :]
            snr3 = self.get_SNR(t, hn, ht, fs)
            snr = (snr1 ** 2 + snr2 ** 2 + snr3 ** 2) ** 0.5
            snr_list.append(snr)
            if zeros_fill:
                snr_list.append(0)
        return snr_list

    def get_SNR(self, t, hn, ht, fs):
        T = t[-1] - t[0]
        NFFT = int(fs / 4)  # should be T*fs/8 to get a better background psd
        psd_window = np.blackman(NFFT)
        NOVL = int(NFFT / 2)
        dt = t[1] - t[0]
        template = ht + ht * 1.j
        datafreq = np.fft.fftfreq(template.size) * fs
        df = np.abs(datafreq[1] - datafreq[0])
        try:
            dwindow = signal.tukey(template.size,
                                   alpha=1. / 8)  # Tukey window preferred, but requires recent scipy version
        except:
            dwindow = signal.blackman(template.size)  # Blackman window OK if Tukey is not available
        template_fft = np.fft.fft(template * dwindow) / fs
        data = hn.copy()
        data_psd, freqs = mlab.psd(data, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL)
        data_fft = np.fft.fft(data * dwindow) / fs
        power_vec = np.interp(np.abs(datafreq), freqs, data_psd)
        optimal = template_fft.conjugate() * data_fft / power_vec
        optimal_time = 2 * np.fft.ifft(optimal) * fs
        sigmasq = 1 * (template_fft * template_fft.conjugate() / power_vec).sum() * df
        sigma = np.sqrt(np.abs(sigmasq))
        SNR_complex = optimal_time / sigma
        peaksample = int(data.size / 2)
        SNR_complex = np.roll(SNR_complex, peaksample)
        # peaksample = int(data.size / 2)
        # SNR_complex = np.roll(SNR_complex, peaksample)
        SNR = abs(SNR_complex)
        indmax = np.argmax(SNR)
        timemax = t[indmax]
        SNRmax = SNR[indmax]
        return SNRmax
