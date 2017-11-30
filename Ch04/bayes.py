# -*- coding: utf-8 -*-

#P(C1|W) > P(C2|W), W in C1
#P(C1|W) < P(C2|W), W in C2  (W = [w1, w2, w3, ..., wi])

#P(Cn|W) = P(W|Cn)*P(Cn)/P(W)
#P(W|Cn) = P(w1|Cn)*P(w2|Cn)*P(w3|Cn)*...*P(wi|Cn)
from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'flea', 
                    'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], 
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'], 
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], 
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        #求两个set的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#输入词汇表和某篇文章
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet: #遍历文章中的每一个单词
        if word in vocabList: #单词在词汇表中
            returnVec[vocabList.index(word)] = 1 #标记出现过该单词(不考虑同一单词重复出现的情况，即认为每个单词的权重相同)
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

'''
listOPosts, listClasses = loadDataSet()
#listClasses = [0,1,0,1,0,1]
myVocabList = createVocabList(listOPosts)
print(myVocabList)
#print(setOfWords2Vec(myVocabList, listOPosts[0]))
#print(setOfWords2Vec(myVocabList, listOPosts[3]))

#标记每篇文章中出现了词汇表中的哪些单词
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
print(trainMat)
'''

def trainNB0(trainMatrix, trainCategory):
    #文档个数 - 5篇
    numTrainDocs = len(trainMatrix)
    #单词个数
    numWords = len(trainMatrix[0])
    #侮辱性文章的比例
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #p0Num = zeros(numWords)
    #p1Num = zeros(numWords)
    #计算p(w0|1)p(w1|1)p(w2|1)...时,如果其中一个概率值为0，那么最后的乘积也为0.为降低这种影响,可以将所有词的出现数初始化为1，并将分母初始化为2.
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    #p0Denom = 0.0; p1Denom = 0.0
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        # 若为侮辱性文章
        if trainCategory[i] == 1:
            #侮辱性文章中，记录每个单词出现的频数
            p1Num += trainMatrix[i]
            #总词数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #在侮辱性文章中，各个单词出现的频率,即[P(wi|C1)]
    #p1Vect = p1Num/p1Denom
    #但为了防止溢出 change to log()
    ##即log([P(wi|C1)])
    p1Vect = log(p1Num/p1Denom)
    #在非侮辱性文章的前提下，各个单词出现的频率,即[P(wi|C0)]
    #p0Vect = p0Num/p0Denom 
    #同样为了防止溢出 change to log()
    #即log([P(wi|C0)])
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

#p0V, p1V, pAb = trainNB0(trainMat, listClasses)


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #log([P(wi|Cn)])
    #log(P(W|Cn)*P(Cn))
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, "classified as: ", classifyNB(thisDoc,p0V,p1V,pAb))

testingNB()
#基于词袋模型的朴素贝叶斯
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
