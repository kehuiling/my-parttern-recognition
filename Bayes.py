import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from sklearn.metrics import roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

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


'''
function:根据训练集计算各个特征的平均值
param:
     train:训练集
     teature_index1:特征值1的column
     feature_index_2:特征值2的column
return:
     mean:各个特征的平均值
'''
def calMean(train,feature_index_1,feature_index_2):
    i = feature_index_1
    j = feature_index_2
    gender_labels = train.iloc[:,-1].value_counts().index
    mean = []
    for k in gender_labels:
        item = train.loc[train.iloc[:,-1]==k,:]
        m = item.iloc[:,i:j+1].mean()
        mean.append(m)
    mean = pd.DataFrame(mean,index=gender_labels)
    return mean


'''
function:计算协方差
param:
      i:特征值1的column
      j:特征值2的column
      data:数据集（男/女）
      mean:各个特征平均值
return:
      cov:（男/女）特征协方差
'''
def calCov(i,j,data,mean):
    data = data.values
    cov = np.mat(np.zeros((2, 2),dtype='float'))
    for h in range(len(data)):
        x = np.mat([data[h, i] - mean[0], data[h, j] - mean[1]])
        cov += x.T * x
    cov /= len(data)-1
    return cov


'''
function:计算概率密度
param:
     data:数据集
     mean:各个特征平均值
     cov:特征协方差
return:
     r:样本落在该类别的概率
     
'''
def calGauss(data,mean,cov):
    data = np.mat(data)
    f1 = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    f2 =float( -1 / 2 * (data - mean) * np.linalg.inv(cov)* (data - mean).T)
    r=float(f1*np.exp(f2))*gender_prod
    return r


'''
function:贝叶斯分类器
param:
     test:测试集
     train:训练集
     teature_index1:特征值1的column
     feature_index_2:特征值2的column
return:
     test:处理后的训练集
     prediction:正样本（女）的概率
     mean_bg:[男特征均值，女特征均值]
     cov_bg:[男特征协方差，女特征协方差]
'''
def bayes_classify(test,train,feature_index_1,feature_index_2):
    i = feature_index_1
    j = feature_index_2
    mean = calMean(train,0,1)
    mean = np.array(mean)
    mean_girl = [mean[0,0],mean[0,1]]
    mean_boy = [mean[1,0],mean[1,1]]
    mean_bg = [mean_girl,mean_boy]
    cov_girl = calCov(0,1,girl_train,mean_girl)
    cov_boy = calCov(0,1,boy_train,mean_boy)
    cov_girl = np.mat(cov_girl)
    cov_boy = np.mat(cov_boy)
    cov_bg = [cov_girl,cov_boy]
    result = []
    prediction = []
    for k in range(test.shape[0]):
        iset = np.array(test.iloc[k,i:j+1])
        iprod_girl = calGauss(iset,mean_girl,cov_girl)
        iprod_boy = calGauss(iset,mean_boy,cov_boy)
        prediction.append(iprod_girl)
        if iprod_girl > iprod_boy:
            result.append('w')
        else:
            result.append('m')
    test['predict']=result
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean()
    print(f'模型预测准确率为{acc}')
    return test,prediction,mean_bg,cov_bg


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


'''
function：绘制ROC曲线
param:
      dataSet:测试数据集
      predic:正例（女）的预测概率
'''
def ROC(dataSet,predic):
    actual = transdata(dataSet)
    predictions = predic
    #得到ROC曲线和曲线下AUC面积
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    #画出ROC曲线
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate,lw=1,label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate')

    plt.show()


'''
function:绘制后验概率3D图、二维面上等高线、决策线
param:
     test:测试集
     mean:男女特征均值
     cov:男女协方差
'''
def draw3D(test,mean,cov):
    #画3D后验概率曲线图
    b_hleft = 100
    b_hright = 200
    b_wleft = 20
    b_wright = 100
    g_hleft = 100
    g_hright = 200
    g_wleft = 20
    g_wright = 100
    b1 = np.linspace(b_hleft, b_hright, 80)
    b2 = np.linspace(b_wleft, b_wright, 80)
    g1 = np.linspace(g_hleft, g_hright, 80)
    g2 = np.linspace(g_wleft, g_wright, 80)
    x1, y1 = np.meshgrid(b1, b2)
    x2, y2 = np.meshgrid(g1, g2)
    z1 = []
    for i in range(len(b1)):
        for j in range(len(b2)):
            data=pd.DataFrame([x1[i][j],y1[i][j]]).iloc[:,0]
            z1.append(calGauss(data,mean[0],cov[0]))
    z1=np.reshape(z1,(len(b1),len(b2)))
    z2 = []
    for i in range(len(g1)):
        for j in range(len(g2)):
            data = pd.DataFrame([x2[i][j], y2[i][j]]).iloc[:, 0]
            z2.append(calGauss(data, mean[1], cov[1]))
    z2 = np.reshape(z2, (len(g1), len(g2)))
    fig=plt.figure(figsize=(15,15))
    ax=fig.add_subplot(2,2,2,projection='3d')
    ax.plot_surface(x1, y1, z1, rstride=1, cstride=1,cmap=plt.cm.hot)
    ax.plot_surface(x2, y2, z2, rstride=1, cstride=1,cmap=plt.cm.cool)
    ax.contour(x1,y1,z1,offset=0.002,colors='black')
    ax.contour(x2,y2,z2,offset=0.002,colors='blue')
    plt.title("The probability density function of the class condition")
    ax.set_xlabel('height',color='r')
    ax.set_ylabel('weight',color='g')
    ax.set_zlabel('posterior probability',color='b')

    #画出投影到二维面上的等高线
    p3=fig.add_subplot(2,2,1)
    C = p3.contour(x1,y1,z1)  # 如果想要在等高线上标出相应的值，需要重新生成一个对象，不能是3d对象
    p3.clabel(C, inline=False, fontsize=10)  # 在等高线上标出对应的z值
    D = p3.contour(x2, y2, z2)  # 如果想要在等高线上标出相应的值，需要重新生成一个对象，不能是3d对象
    p3.clabel(D, inline=False, fontsize=10)  # 在等高线上标出对应的z值

    #画出决策线
    Colors=[]
    #不同性别用不同颜色区分（男：蓝，女：橘）
    for i in range(test.shape[0]):
        m=test.iloc[i,-2]
        if m == 'm':
            Colors.append('blue')
        if m== 'w':
            Colors.append('orange')
            
    cmap_light =ListedColormap(['#AAAAFF','#F0E68C'])
    z3=[]
    for i in range(len(b1)):
        for j in range(len(b2)):
            data=pd.DataFrame([x1[i][j],y1[i][j]]).iloc[:,0]
            g_r=calGauss(data,mean[0],cov[0])
            b_r=calGauss(data,mean[1],cov[1])
            if g_r>=b_r:
                z3.append(1)
            else:
                z3.append(0)
    z3=np.reshape(z3,(len(b1),len(b2)))
    p4=fig.add_subplot(2,2,3)
    p4.contourf(x1,y1,z3,cmap=cmap_light)
    x = test.iloc[:, 0].tolist()
    y = test.iloc[:, 1].tolist()
    p4.scatter(x,y,c=Colors)
    plt.ioff()
    plt.show()

dataSet = pd.read_table(path,header=None,sep=':')
girlMat,boyMat = genderSplit(dataSet)
girl_train,girl_test = randSplit(girlMat,rate)
boy_train,boy_test = randSplit(boyMat,rate)
train = mergeMat(girl_train,boy_train)
test = mergeMat(girl_test,boy_test)
test_height_weight,predic,mean_bg,cov_bg = bayes_classify(test,train,0,1)
ROC(test_height_weight,predic)
draw3D(test_height_weight,mean_bg,cov_bg)