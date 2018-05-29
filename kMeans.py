'''
Created on Feb 16, 2011
Modify on Mar 27, 2018
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington ---++++---YM 
'''
from numpy import *
import numpy as np


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        fltLine = []
        curLine = line.strip().split('\t')  # 读取的数据是坐标形式          
        fltLine = [float(curLine[0]),float(curLine[1])]
        dataMat.append(fltLine)
        # fltLine = map(float,curLine) # map all elements to float()
        # 为什么会显示map类型呢？？？？？？？？？？map不是函数应用吗
        # print(curLine[0])
        # print(type(fltLine))
    return mat(dataMat)
  
  
# calculate Euclidean distance  
def euclDistance(vector1, vector2):  
    return sqrt(sum(power(vector2 - vector1, 2)))  #求这两个矩阵的距离，vector1、2均为矩阵
  
# init centroids with random samples  
#在样本集中随机选取k个样本点作为初始质心
def initCentroids(dataSet, k):  
    numSamples, dim = dataSet.shape   #矩阵的行数、列数 
    centroids = zeros((k, dim))         #感觉要不要你都可以
    for i in range(k):  
        index = int(random.uniform(0, numSamples))  #随机产生一个浮点数，然后将其转化为int型
        centroids[i,:] = dataSet[index, :]  
    return centroids
  
# k-means cluster 
#dataSet为一个矩阵
#k为将dataSet矩阵中的样本分成k个类 
def kMeans(dataSet, k):  
# def kMeans(dataSet, k, distMeas=euclDistance, createCent=initCentroids):
    numSamples = dataSet.shape[0]  #读取矩阵dataSet的第一维度的长度,即获得有多少个样本数据
    # first column stores which cluster this sample belongs to,  
    # second column stores the error between this sample and its centroid  
    clusterAssment = mat(zeros((numSamples, 2)))  #得到一个N*2的零矩阵
    clusterChanged = True  
  
    ## step 1: init centroids  
    centroids = initCentroids(dataSet, k)  #在样本集中随机选取k个样本点作为初始质心
  
    while clusterChanged:  
        clusterChanged = False  
        ## for each sample  
        for i in range(numSamples):  #range
            minDist  = 100000.0  
            minIndex = 0  
            ## for each centroid  
            ## step 2: find the centroid who is closest  
            #计算每个样本点与质点之间的距离，将其归内到距离最小的那一簇
            for j in range(k):  
                distance = euclDistance(centroids[j, :], dataSet[i, :])  
                if distance < minDist:  
                    minDist  = distance  
                    minIndex = j                
            ## step 3: update its cluster 
            #k个簇里面与第i个样本距离最小的的标号和距离保存在clusterAssment中
            #若所有的样本不在变化，则退出while循环
            if clusterAssment[i, 0] != minIndex:  
                clusterChanged = True  
                clusterAssment[i, :] = minIndex, minDist**2  #两个**表示的是minDist的平方
  
        ## step 4: update centroids  
        for j in range(k):  
            #clusterAssment[:,0].A==j是找出矩阵clusterAssment中第一列元素中等于j的行的下标，返回的是一个以array的列表，第一个array为等于j的下标
            pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]] #将dataSet矩阵中相对应的样本提取出来 
            centroids[j, :] = mean(pointsInCluster, axis = 0)  #计算标注为j的所有样本的平均值
  
    print ('Congratulations, cluster complete!')  
    print(type(centroids),type(clusterAssment))
    return mat(centroids), clusterAssment  
  


def biKmeans(dataSet, k):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = euclDistance(mat(centroid0), dataSet[j])**2
    while (len(centList) < k):
        lowestSSE = inf    # 尽量使得SSE最小
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]# get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment


import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(datMat,numClust):
    # datList = []
    # for line in open('places.txt').readlines():
    #     lineArr = line.split('\t')
    #     datList.append([float(lineArr[4]), float(lineArr[3])])
    # datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust)  # 用的是二分kMeans聚类
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == '__main__':    
    # print(type(datmat))
    datmat = loadDataSet('testSet.txt')
    # print(datmat[:][0])# 第一列的最小值    
    # print(distEclud(datmat[0],datmat[1]))  
    #---------------------------  
    # datmat = mat(np.random.rand(100,2))
    # print(type(datmat))
    clusterClubs(datmat,4) 
    # myCentroids,clustAssing = kMeans(datmat,4)
