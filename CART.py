import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

path='C:\\Users\\WIN10\\Desktop\\ex00.txt'
ex00= pd.read_table(path,header=None)
plt.scatter(ex00.iloc[:,0].values,ex00.iloc[:,1].values)
plt.show()


"""
功能：根据特征切分数据集合
参数：
        dataSet:原始数据集
        feature:待切分特征得索引
        value:该特征得值
返回:
        mat0:切分的数据集合0
        mat1:切分的数据集合1
"""
def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet.loc[dataSet.iloc[:,feature]>value,:]
    mat1=dataSet.loc[dataSet.iloc[:,feature]<=value,:]
    return mat0,mat1


'''
功能：计算切分误差函数
参数：dataSet:数据集
'''
def errType(dataSet):
    #计算总方差：均方差*样本数
    var= dataSet.iloc[:,-1].var() *dataSet.shape[0]
    return var


'''
功能：生成叶节点函数
**************************************************************************************
*** 当最佳切分函数确定不再对数据进行切分时，将调用该函数来得到叶节点的模型。在回归树中，该模型其实 ***
*** 就是目标变量的均值。（模型树是通过一个线性函数）                                       ***
**************************************************************************************
'''
def leafType(dataSet):
    leaf=dataSet.iloc[:,-1].mean()
    return leaf


'''
功能：找到数据的最佳二元切分方式
参数：
     dataSet:待切分数据
     leafType:生成叶节点函数
     errType:误差估计函数
     ops:用户定义的参数构成的元组
返回：
     spInd:最佳切分特征的索引
     spVal:最佳切分特征的值
'''

def chooseBestSplit(dataSet, leafType=leafType, errType=errType, ops = (1,4)):
    #tols:允许的误差下降值   tolN:切分的最小样本数
    tolS = ops[0]; tolN = ops[1]
    #如果当前所有值都相等则退出
    if len(set(dataSet.iloc[:,-1].values)) == 1:
        return None, leafType(dataSet)
    #统计数据集的行数和列数
    m, n = dataSet.shape
    #默认最后一个特征为最佳切分，计算其误差估计
    S = errType(dataSet)
    #最佳误差，最佳特征索引，最佳特征值
    bestS = np.inf; bestIndex = 0; bestValue = 0
    #遍历所有特征列
    for featIndex in range(n - 1):
        colval= set(dataSet.iloc[:,featIndex].values)
        #遍历所有特征值
        for splitVal in colval:
            #切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            #若数据数小于最小样本数则退出
            if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
                continue
            #计算误差估计
            newS = errType(mat0) + errType(mat1)
            #若误差小于最佳误差，更新特征索引和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #若误差减小不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    #根据最佳切分特征切分数据集
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #若切分后的数据集很小则退出
    if (mat0.shape[0] < tolN) or (mat1.shape[0] < tolN):
        return None, leafType(dataSet)
    return bestIndex, bestValue


'''
功能：构建树
参数：
     dataSet:待切分数据
     leafType:生成叶节点函数
     errType:误差估计函数
     ops:用户定义的参数构成的元组
返回：
     retree:构建的回归树
     
'''
def createTree(dataSet, leafType = leafType, errType = errType, ops = (1, 4)):
    #选择最佳切分特征进行切分
    col,val=chooseBestSplit(dataSet, leafType, errType, ops)
    #若没有，则返回特征值
    if col==None:
        return val
    #回归树
    retTree = {}
    retTree['spInd'] = col
    retTree['spVal'] = val
    #分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, col, val)
    #创建左子树和右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


'''
功能：根据回归树预测数据值
参数：
     dataSet:数据集
     tree:回归树
     
'''

def tree_predict(dataSet,tree):
    if type(tree) is not dict:
        return tree
    spInd,spVal=tree['spInd'],tree['spVal']
    if dataSet[spInd]>spVal:
        sub_tree=tree['left']
    else:
        sub_tree=tree['right']
    return tree_predict(dataSet,sub_tree)


plt.scatter(ex00.iloc[:,0].values,ex00.iloc[:,1].values)
tree=createTree(ex00)
x=np.linspace(0,1,50)
y=[tree_predict([i],tree)for i in x]
plt.plot(x,y,c='r')
plt.show()




