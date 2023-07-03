import os
import shutil
from os import listdir  # 使用listdir模块，用于访问本地文件
from random import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from sklearn import neighbors

class CharacterRecognition():

    def rename(self,path,flag):
        # 批量改名,并设置标签，zhong-0 guo-1
        filelist = os.listdir(path)
        total_num = len(filelist)
        i = 0
        for item in filelist:
            if item.endswith('.bmp'):
                src = os.path.join(os.path.abspath(path), item)
                dst = os.path.join(os.path.abspath(path), str(flag) + '_' + str(i) + '.bmp')
                try:
                    os.rename(src, dst)
                    i += 1
                except:
                    continue
            print('total %d to rename & converted %d bmp' % (total_num, i))

    def img2vector(self,fileName):
        retMat = np.zeros([14700], int)
        img = Image.open(fileName, "r")
        array = np.array(img.getdata())
        for i in range(len(array)):  # 遍历文件所有行
            for j in range(3):
                retMat[i * 3 + j] = array[i][j]
        return retMat

    def read_data(self,path, split_rate):
        #从数据集中随机选取split_rate*nums作为测试集，剩下的作为训练集,split_rate=0.25
        fileList = os.listdir(path)
        total_num = len(fileList)
        dataSet = np.zeros([total_num, 14700], int)  # 用于存放所有的数字文件
        labels = np.zeros([total_num], str)  # 用于存放对应的标签(与神经网络的不同)
        for i in range(total_num):
            filePath = fileList[i]  # 获取文件名称/路径
            digit = str(filePath.split('_')[0])  # 通过文件名获取标签
            labels[i] = digit  # 直接存放数字，并非one-hot向量
            dataSet[i] = self.img2vector(path + '/' + filePath)  # 读取文件内容

        # split_rate*total_num
        index = [i for i in range(int(split_rate * total_num))]
        index1 = [i for i in range(int(split_rate * total_num), total_num)]
        np.random.shuffle(index)
        np.random.shuffle(index1)
        test_dataSet = dataSet[index]
        test_labels = labels[index]
        train_dataSet = dataSet[index1]
        train_labels = labels[index1]
        return train_dataSet, train_labels, test_dataSet, test_labels

    def suitable_k(self, data, label):
        # 作图选择合适的K：8
        k_range = range(1, 10)
        k_error = []  # 错误率
        for k in k_range:
            knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=k)
            # algorithm设置分类器算法为kd_tree，改变n_neighbors的值改变所需要查询的最近邻数目
            knn.fit(data, label)
            res = knn.predict(data)  # 对测试集进行预测
            error_num = np.sum(res != label)  # 统计分类错误的数目
            num = len(data)  # 测试集的数目
            k_error.append(error_num / float(num))

        plt.plot(k_range, k_error)
        plt.xlabel('Value of K in KNN')
        plt.ylabel('Error')
        plt.show()

    def generate_model(self, data, label):
        knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=8)
        knn.fit(data, label)
        joblib.dump(knn, './model.pkl')  # 将模型保存到本地


if __name__=='__main__':
    demo = CharacterRecognition()
    path = "data_set"
    path1 = path + "/guo"
    path2 = path + "/zhong"
    # demo.rename(path1,1)
    # demo.rename(path2,0)
    train_dataSet1, train_labels1, test_dataSet1, test_labels1 = demo.read_data(path1,0.25)
    train_dataSet2, train_labels2, test_dataSet2, test_labels2 = demo.read_data(path2, 0.25)

    train_labels = list(train_labels1)+list(train_labels2)
    train_dataSet = np.append(train_dataSet1,train_dataSet2,axis=0)
    # print(len(train_dataSet))

    labels = list(test_labels1) + list(test_labels2)
    dataSet = np.append(test_dataSet1, test_dataSet2, axis=0)

    # demo.suitable_k(train_dataSet, train_labels)
    # demo.generate_model(train_dataSet, train_labels)

    models_path = './model.pkl'
    clf = joblib.load(models_path)

    # 预测
    path3 = path + "/predict"
    filelist = os.listdir(path3)
    total_num = len(filelist)
    input = []
    print("-----------手写汉字识别-----------")
    print("路径下的图片总数为：", total_num)

    for i in range(total_num):
        ds = demo.img2vector(path3 + "/" + filelist[i])
        input.append(ds)
    res = clf.predict(input)
    for i in range(len(res)):
        if res[i] == '0':
            print("图片中的汉字为：中")
        if res[i] == '1':
            print("图片中的汉字为：国")

    print("预测结束")


















