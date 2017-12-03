# -*- coding: utf-8 -*-
from numpy import *

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

#dataMatIn是100*3矩阵, classLabels是1*100矩阵
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    #矩阵的转置
    labelMat = mat(classLabels).transpose()
    #m = 100, n = 3
    m, n = shape(dataMatrix)
    #步长
    alpha = 0.001
    #迭代次数
    maxCycles = 500
    #weights是3*1矩阵
    weights = ones((n,1))
    #矩阵相乘
    for k in range(maxCycles):
        #h是一个100*1的列向量
        h = sigmoid(dataMatrix*weights)
        #error是100*1的列向量
        error = (labelMat - h)
        #3*100矩阵乘100*1矩阵
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

'''
dataArr, labelMat = loadDataSet()
print(gradAscent(dataArr,labelMat))
'''

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s = 30, c = 'red', marker = 's')
    ax.scatter(xcord2, ycord2, s = 30, c = 'green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

dataArr, labelMat = loadDataSet()    
weights = gradAscent(dataArr,labelMat)
plotBestFit(weights)