"""
Author: Lancher
Date: 2022.6.17
"""
from numpy import *
import operator


# 读取手写数据
def readData(filename):
    labels = []
    dataset = []
    with open(filename, 'r') as f:
        sts = f.readlines()
        f.close()
    for st in sts:
        labels.append(st.split(',')[0])
        dataset.append(list(map(int, st.split(',')[1:])))
    dataset = array(dataset)
    return labels, dataset


# 实现算法
def classify0(inX, dataset, labels, k):
    datasetSize = dataset.shape[0]  # 获取维数
    diffMat = tile(inX, (datasetSize, 1)) - dataset  # 将输入向量扩展成和数据集维数相同的矩阵，然后计算差值
    sqDiffMat = diffMat ** 2  # 差值矩阵各项平方
    sqDistance = sum(sqDiffMat, axis=1)  # 平方后计算每项之和
    distances = sqDistance ** 0.5  # 开方
    sortedDistIndicies = distances.argsort()  # 返回数组从小到大的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 统计
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount


def handwriting():
    l, d = readData('mnist_train.csv')
    l1, inX = readData('mnist1.csv')
    passCount = []
    for inx in range(len(inX)):
        res = classify0(inX[inx], d, l, 10)[0][0]
        print('The result is ', res, '. The correct answer is ', l1[inx])
        passCount.append(1 if res == l1[inx] else 0)
    print('rate:', sum(passCount) / len(passCount))


if __name__ == '__main__':
    handwriting()
