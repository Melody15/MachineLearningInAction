# -*- coding: utf-8 -*-

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

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print(myVocabList)
print(setOfWords2Vec(myVocabList, listOPosts[0]))
print(setOfWords2Vec(myVocabList, listOPosts[3]))