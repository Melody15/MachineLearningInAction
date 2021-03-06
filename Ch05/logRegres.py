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
    return longfloat(1.0/(1+exp(-inX)))

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
        #计算真实类别与预测类别的差值，接下来就是按照该差值的方向调整回归系数
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
    #矩阵通过getA()方法可以将自身返回成一个n维数组对象
    #print(wei)
    #print(wei[1])
    #wei[1] is [[ 0.48007329]]
    #weights = wei.getA()
    #print(weights)
    #print(weights[1])
    #weights[1] is [ 0.48007329]
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

'''
dataArr, labelMat = loadDataSet()    
#weights is a matrix
weights = gradAscent(dataArr,labelMat)
print(weights)
plotBestFit(weights)
'''

def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

'''
weights = stocGradAscent0(array(dataArr),labelMat)
print(weights)
plotBestFit(weights)
'''

def stocGradAscent1(dataMatrix, classLabels, numIter = 150):
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
    return weights
    
'''
weights = stocGradAscent0(array(dataArr),labelMat)
print(weights)
plotBestFit(weights)
'''

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

multiTest()
