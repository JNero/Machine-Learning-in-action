from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
 	vocabSet=set([])
 	for document in dataSet:
 		vocabSet=vocabSet|set(document)
 	return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
 	returnVec=[0]*len(vocabList)
 	for word in inputSet:
 		if word in vocabList:
 			returnVec[vocabList.index(word)]=1
 		else:
 			pirnt("the word ",word,"is not in my Vocabulary")
 	return returnVec

def trainNB0(trainMartix,tarinCategory):
	numTrainDocs=len(trainMartix)
	numWords=len(trainMartix[0])
	pAbusive=sum(tarinCategory)/float(numTrainDocs)
	p0Num=zeros(numWords)
	p1Num=zeros(numWords)
	p0Denom=0.0
	p1Denom=0.0
	for i  in range(numTrainDocs):
		if tarinCategory[i]==1:
			p1Num+=trainMartix[i]
			p1Denom+=trainMartix[i]
		else:
			p0Num+=trainMartix[i]
			p0Denom+=sum(trainMartix[i])
	p1Vect=p1Num/p1Denom
	p0Vect=p0Num/p0Denom
	return p0Vect,p1Vect,pAbusive