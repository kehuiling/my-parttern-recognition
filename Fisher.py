import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import roc_curve, auc
from matplotlib.colors import ListedColormap
import Bayes as bs

'''
function:用训练样本集建立Bayes 分类器，用测试样本数据对该分类器进行测试。调
         整特征、分类器等方面的一些因素，考察它们对分类器性能的影响
param:
      path:原始数据集路径
      rate:训练集所占原始数据集比重
      gender_prod:先验概率（每种性别出现的概率）
'''
path = 'C:\\Users\\WIN10\\Desktop\\train.txt'
rate = 0.9
gender_prod = 0.55

'''
function:将数据集切分成男女两类
param:
      dataSet:原始数据集
return:
      girlMat:女
      boyMat:男
'''
def genderSplit(dataSet):
    girlMat = dataSet.loc[dataSet.iloc[:,-1]=='w',:]
    boyMat = dataSet.loc[dataSet.iloc[:,-1]=='m',:]
    girlMat.index = range(girlMat.shape[0])
    boyMat.index = range(boyMat.shape[0])
    return girlMat,boyMat


'''
function:随机切分训练集和测试集
param:
      dataSet:原始数据集
      rate:训练集所占比例
return:
      train:训练集
      test:测试集
'''
def randSplit(dataSet,rate):
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
function:合并数据集
param:
      girlMart:女数据集
      boyMat:男数据集
return：
       Mat:合并后的数据集

'''
def mergeMat(girlMat,boyMat):
    girlnum = int(girlMat.shape[0])
    boynum = int(boyMat.shape[0])
    Mat1 = girlMat.loc[range(girlnum),:]
    Mat2 = boyMat.loc[range(boynum),:]
    list = []
    list.append(Mat1)
    list.append(Mat2)
    Mat = pd.concat(list)
    Mat.index = range(Mat.shape[0])
    return Mat


def calFisher(feature_index1,feature_index2,girlMat,boyMat):
    girl_mean = np.array(girlMat.iloc[:,feature_index1:feature_index2+1].mean())
    boy_mean = np.array(boyMat.iloc[:,feature_index1:feature_index2+1].mean())
    girlMat = np.array(girlMat)
    boyMat = np.array(boyMat)
    cov = np.mat(np.zeros((2, 2), dtype='float'))
    for i in range(girlMat.shape[0]):
        x = np.mat([girlMat[i,feature_index1] - girl_mean[0],girlMat[i,feature_index2] - girl_mean[1]])
        cov += x.T * x
    for j in range(boyMat.shape[0]):
        x = np.mat([boyMat[j,feature_index1] - boy_mean[0],boyMat[j,feature_index2] - boy_mean[1]])
        cov += x.T * x
    Sw = np.linalg.inv(cov)
    w = Sw * (np.mat(girl_mean - boy_mean).T)
    m1 = (w.T * np.mat(girl_mean).T)
    m2 = (w.T * np.mat(boy_mean).T)
    return w.T, m1[0, 0], m2[0, 0]


def fisher_classify(girl_train,boy_train,test,feature_index1,feature_index2):
    i = feature_index1
    j = feature_index2
    w,m1,m2 = calFisher(i,j,girl_train,boy_train)
    w0 = gender_prod * m1 + (1 - gender_prod) * m2
    result = []
    prediction = []
    for k in range(test.shape[0]):
        y = (w * np.mat(test.iloc[k,i:j+1].values).T)-w0
        if y > 0:
            result.append('w')
        else:
            result.append('m')
        prediction.append(y[0, 0])
    test['predict']=result
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean()
    print(f'模型预测准确率为{acc}')
    return test,prediction


'''
function:设置正例，此处为女
param:
     dataSet:数据集
return:
      y_true:存放正例的list
'''
def transdata(dataSet):
    y_true = []
    for i in range(dataSet.shape[0]):
        if dataSet.iloc[i,-2] == 'w':
            y_true.append(1)
        else:
            y_true.append(0)
    return y_true



def draw(test,predic,girlMat,boyMat,girl_train,boy_train,i,j):
    actual = transdata(test)
    predictions = predic
    #得到ROC曲线和曲线下AUC面积
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    #画出ROC曲线
    plt.subplot(221)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate,lw=1,label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')


    plt.subplot(222)
    w,m1,m2 = calFisher(i,j,girl_train,boy_train)
    w0 = gender_prod * m1 + (1 - gender_prod) * m2
    x = girlMat.iloc[:, i].append(boyMat.iloc[:, i])
    y = girlMat.iloc[:, j].append(boyMat.iloc[:, j])
    x_min, x_max = x.min() - 1.5, x.max() + 1.5
    y_min, y_max = y.min() - 1.5, y.max() + 1.5
    h = 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = np.zeros(xx.shape, dtype='int')
    for k in range(xx.shape[0]):
        for g in range(xx.shape[1]):
            data = [xx[k, g], yy[k, g]]
            y = (w * np.mat(data).T) - w0
            if y > 0:
                z[k, g] = 1
    Colors=[]
    #不同性别用不同颜色区分（男：蓝，女：橘）
    for t in range(test.shape[0]):
        m=test.iloc[t,-2]
        if m == 'm':
            Colors.append('blue')
        if m== 'w':
            Colors.append('orange')
            
    cmap_light =ListedColormap(['#AAAAFF','#F0E68C'])
    plt.pcolormesh(xx, yy, z,cmap=cmap_light)
    x1 = test.iloc[:, i].tolist()
    y1 = test.iloc[:, j].tolist()
    plt.scatter(x1, y1, c=Colors)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('boundary line of Fisher')

    plt.subplot(223)
    z2 = np.zeros(xx.shape, dtype='int')
    girl_mean = np.array(bs.calMean(girl_train,i,j))
    print(girl_mean)
    mean_g = [girl_mean[0,0],girl_mean[0,1]]
    boy_mean = np.array(bs.calMean(boy_train,i,j))
    mean_b = [boy_mean[0,0],boy_mean[0,1]]
    mean_bg = [mean_g,mean_b]
    girl_cov = np.mat(bs.calCov(i,j,girl_train,mean_g))
    boy_cov = np.mat(bs.calCov(i,j,girl_train,mean_b))
    cov_bg = [girl_cov,boy_cov]
    print(mean_bg)
    print(cov_bg)
    for k in range(xx.shape[0]):
        for g in range(xx.shape[1]):
            data = pd.DataFrame([xx[k, g], yy[k, g]]).iloc[:,0]
            print(data)
            g_r=bs.calGauss(data,mean_bg[0],cov_bg[0])
            b_r=bs.calGauss(data,mean_bg[1],cov_bg[1])
            if g_r>=b_r:
                z2[k,g] = 1
    plt.contourf(xx, yy, z2, cmap=cmap_light)
    plt.scatter(x1, y1, c=Colors)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('boundary line of Bayes')

    plt.subplot(224)
    plt.ylim(35,90)
    # plt.contour(xx, yy, z)
    x3 = np.arange(x_min + 8, x_max - 8, h)
    y4 = (w0 - w[0, 0] * x3) / w[0, 1]
    y5=w[0,1]/w[0,0]*x3
    plt.plot(x3,y4,color='green')
    plt.plot(x3,y5,color='green')
    plt.contour(xx, yy, z2,color='y')
    plt.scatter(x1, y1, c=Colors)
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.title('boundary line of two classfiers')

    plt.show()


dataSet = pd.read_table(path,header=None,sep=':')
girlMat,boyMat = genderSplit(dataSet)
girl_train,girl_test = randSplit(girlMat,rate)
boy_train,boy_test = randSplit(boyMat,rate)
train = mergeMat(girl_train,boy_train)
test = mergeMat(girl_test,boy_test)
test_hw,prediction = fisher_classify(girl_train,boy_train,test,0,1)
draw(test_hw,prediction,girlMat,boyMat,girl_train,boy_train,0,1)
