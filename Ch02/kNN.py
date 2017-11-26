from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0,1.1], [1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #4
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    #duplicate the inX into the format of dataSet
    sqDiffMat = diffMat**2
    #^2 each element
    sqDistances = sqDiffMat.sum(axis=1)
    #calculate the sum of each line's characteristic value
    distances = sqDistances**0.5
    #^0.5 each sum
    sortedDistIndicies = distances.argsort()
    #sort the index by the distances
    classCount = {}
    for i in range(k): #first kth elements
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        #get() - Return the value for key if key is in the dictionary, else default. If default is not given, it defaults to None, so that this method never raises a KeyError.
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    #items() - Return a new view of the dictionary’s items ((key, value) pairs).
    #itemgetter() - Return a callable object that fetches item from its operand using the operand’s __getitem__() method. 
    return sortedClassCount[0][0]

'''
group,labels = createDataSet()
print(classify0([1,1],group,labels,3))
'''

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip() # clear the enter
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
'''
datingDataMat, datingLabels = file2matrix("datingTestSet.txt")


fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()
'''

def autoNorm(dataSet):
    minVals = dataSet.min(0) #0 means select each minimum from the rows
    maxVals = dataSet.max(0)
    # minVals&maxVals are 1*3 matrix
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0] #line
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

'''
norMat, ranges, minVals = autoNorm(datingDataMat)
'''

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix("datingTestSet.txt")
    norMat, ranges, minVals = autoNorm(datingDataMat)
    m = norMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],3)
        print("the classifier came back with %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))