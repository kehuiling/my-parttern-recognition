import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random


'''
function:同时采用身高、体重和鞋码数据作为特征，设计k近邻方法性别分类器，
         将该分类器应用到训练和测试样本，考察训练和测试错误情况
param:
      path:原始数据集路径
      rate:训练集所占原始数据集比重
      k:k-近邻算法参数，选择距离最小的k个点
'''


path='C:\\Users\\WIN10\\Desktop\\train.txt'
rate=0.9
k=4


'''
function:讲txt中的数据导入字典
return:
       rowdata:存放原始数据集的字典
'''
def datain(txtpath):
    rowdata = {'1height':[],'2weight':[],'3footsize':[],'4gender':[]}
    with open(txtpath,'r') as f:
        txt=f.read()
        for t in txt.split():
            hvalue,wvalue,fvalue,gvalue=t.split(':',3)
            rowdata['1height'].append(float(hvalue))
            rowdata['2weight'].append(float(wvalue))
            rowdata['3footsize'].append(float(fvalue))
            rowdata['4gender'].append(gvalue)
    return rowdata
 

'''
function:数据归一化
param:
      dataSet:数据集
return:
      normSet:归一化后的数据集
'''
def minmax(dataSet):
    minDf=dataSet.min()
    maxDf=dataSet.max()
    normSet=(dataSet - minDf)/(maxDf - minDf)
    return normSet


'''
function:切分训练集和测试集
param:
      dataSet:原始数据集
      rate:训练集所占比例
return:
      train:训练集
      test:测试集
'''

def randSplit(dataSet, rate):
    l = list(dataSet.index)                   #提取出索引
    random.shuffle(l)                         #随机打乱索引
    dataSet.index = l                         #将打乱后的索引重新赋值给原数据集
    n = dataSet.shape[0]                      #总行数
    m = int(n * rate)                         #训练集的数量
    train = dataSet.loc[range(m), :]          #提取前m个做训练集
    test = dataSet.loc[range(m, n), :]        #剩下的作为测试集
    dataSet.index = range(dataSet.shape[0])   #更新原数据集的索引
    test.index = range(test.shape[0])         #更新测试集索引
    return train, test


'''
function:KNN算法
param:
      train:训练集
      test:测试集
      k:k-近邻算法参数
return:
      test:预测好分类的测试集（加了一列“预测结果（predict)")
'''
def KNN(train,test,k):
    n = train.shape[1] - 1
    m = test.shape[0]
    result = []
    for i in range(m):
        dist = list((((train.iloc[:, :n] - test.iloc[i, :n]) ** 2).sum(1))**5)
        dist_l = pd.DataFrame({'dist':dist,'gender':(train.iloc[:, n])})
        dr = dist_l.sort_values(by = 'dist')[: k]
        re = dr.loc[:, 'gender'].value_counts()
        result.append(re.index[0])
    result = pd.Series(result)
    test['predict'] = result
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean()
    print(f'模型预测准确率为{acc}')
    return test


def KNN_predict(inX,dataSet,k):
    result=[]
    dist = list((((dataSet.iloc[:,0:3]-inX)**2).sum(1))**0.5)
    dist_l = pd.DataFrame({'dist':dist,'gender':(dataSet.iloc[:, 3])})
    dr = dist_l.sort_values(by = 'dist')[: k]
    re = dr.loc[:, 'gender'].value_counts()
    result.append(re.index[0])
    print(result)
    return result



'''
function:创建散点图
param:
     dataSet:数据集
'''
def colors(dataSet):
    Colors=[]

    #不同性别用不同颜色区分（男：蓝，女：橘）
    for i in range(dataSet.shape[0]):
        m=dataSet.iloc[i,-1]
        if m == 'm':
            Colors.append('blue')
        if m== 'w':
            Colors.append('orange')        



    #绘制两两特征间的散点图
    plt.rcParams['font.sans-serif']=['simhei']
    pl=plt.figure(figsize=(12,8))
    fig1=pl.add_subplot(221)
    plt.scatter(dataSet.iloc[:,0],dataSet.iloc[:,1],marker='.',c=Colors)
    plt.xlabel('height')
    plt.ylabel('weight')

    fig2=pl.add_subplot(222)
    plt.scatter(dataSet.iloc[:,1],dataSet.iloc[:,2],marker='.',c=Colors)
    plt.xlabel('weight')
    plt.ylabel('footsize')

    fig3=pl.add_subplot(223)
    plt.scatter(dataSet.iloc[:,0],dataSet.iloc[:,2],marker='.',c=Colors)
    plt.xlabel('height')
    plt.ylabel('footsize')

    plt.show()

gender_data=pd.DataFrame(datain(path))
dataSet=pd.concat([minmax(gender_data.iloc[:, :3]), gender_data.iloc[:,3]], axis=1)
train,test=randSplit(dataSet,rate)

KNN(train,test,k)
colors(train)
colors(test)

