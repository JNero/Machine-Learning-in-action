from numpy import *
import operator
import os
#import importlib
# importlib.reload(kNN)
from numpy.ma import array, zeros


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndcies = distances.argsort()
    classCount = {}
    for i in range(k):
        #        xi=sortedDistIndcies[i]
        #        print(xi)
        #        print(labels[xi])
        voteIlabel = labels[sortedDistIndcies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2martix(filename):
        #  use Regular expression "\s[3]\s" to "\tlargeDoses\n" to replace
        #   "\s[2]\s" to "\tsmallDoses\n"
        # "\s[1]\s" to "\tdidntLike\n"
    labels2 = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
#    print(numberOfLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(labels2[listFromLine[-1]])
        index += 1
    return returnMat, classLabelVector


def autonormal(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2martix('datingTestSet.txt')
    normMat, ranges, minVals = autonormal(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(
            normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: " + classifierResult +
              "the real answer is: ", datingLabels[i])
        if classifier != datingLabels:
            errorCount += 1.0
        print("the total error rate is", (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("Game: "))
    ffMiles = float(input("Fly Miles: "))
    iceCream = float(input("Eat Icecream: "))
    datingDataMat, datingLabels = file2martix('datingTestSet2.txt')
    normMat, ranges, minVals = autonormal(datingDataMat)
    inarr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify(
        (inarr - minVals) / ranges,
        normMat,
        datingLabels,
        3)
    print("you will probably lis this person: ",
          resultList[classifierResult - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
#    print(m)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
#        print(classNumStr)
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/' + fileNameStr)
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    ###
#        print(hwLabels)
#        print("####")
#        print(len(hwLabels))
    ###
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/' + fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print(
            "the classifier came back with: " +
            str(classifierResult) +
            "the real answer is: " +
            str(classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
        print("the total number of errors is " + str(errorCount))
        print("the total error rate is" + str(errorCount / float(mTest)))
