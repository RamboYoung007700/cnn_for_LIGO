# -*-coding:utf-8-*-
# 分类程序(for LIGO DATA)v1.0，杨楠，2018、10、05
# 创建神经网络
# 建议使用pycharm运行本程序。
"""
使用方法：
直接运行就可以了。根据需要来创建神经网络。
这个网络会被保存，但是不会被训练。训练要在train_network.py中进行
"""

from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten

# 建立神经网络，keras库采用自己创建模型，模型中加各种层来得到需要训练的神经网络
model = Sequential()
# input: 3x4096 images with 1 channels -> (3, 4096, 1) tensors.

model.add(Convolution2D(16, (1, 16), activation='relu', batch_input_shape=(1000, 3, 4096, 1)))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Convolution2D(32, (1, 8), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Convolution2D(64, (1, 8), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 4)))
model.add(Convolution2D(128, (1, 8), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 4)))
model.add(Convolution2D(256, (3, 8), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 4)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# 神经网络建立完毕

# 保存神经网络
model.save('cnn_for_LIGO_0.h5')
