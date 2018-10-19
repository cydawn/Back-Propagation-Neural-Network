import math
import random
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import float32
##定义sinx/x函数
def sinx(a,b):
    if (a != np.float64(0)) & (b != np.float64(0)):
       return (np.sin(a)*np.sin(b))/(a*b)
    elif (a != np.float64(0)) & (b ==np.float64(0)):
        return np.sin(a)/a
    elif (a == np.float64(0)) & (b != np.float64(0)):
        return np.sin(b)/b
    else :
        return 1.0
##定义训练样本和检验样本
##X_train1和Y_train1为问题一的训练样本，X_test1和Y_test1的检测样本
##X_train2和Y_train2为问题一的训练样本，X_test2和Y_test2的检测样本
X_train1 = np.array([[[i*np.pi*2*1.0/8]] for i in range(9)])
X_test1 = np.array([[[i*np.pi*2*1.0/360]] for i in range(361)])
Y_train1 = np.array([[np.sin(i)] for i in X_train1])
Y_test1 = np.array([np.sin(i) for i in X_test1])
X_train2 = np.zeros(shape=(11,11,2))
Y_train2 = np.zeros(shape=(11,11))
for i in range(11):
    for j in range(11):
        X_train2[i][j] = (-10.0+20.0*i/10,-10.0+20.0*j/10)
        Y_train2[i][j] = sinx(X_train2[i][j][0],X_train2[i][j][1])
X_test2 = np.zeros(shape=(21,21,2))
Y_test2 = np.zeros(shape=(21,21))
for i in range(21):
    for j in range(21):
        X_test2[i][j] = (-10.0+20.0*i/20,-10.0+20.0*j/20)
        Y_test2[i][j] = sinx(X_test2[i][j][0],X_test2[i][j][1])
Y_out = []  ##检测样本输出
error_out = []    ##误差输出，用于绘制误差曲线
##学习率变步长
def rate(learning_rate,error):
    if error > 0.3:
        rate = learning_rate
    elif error > 0.1:
        rate = 2 * learning_rate
    else :
        rate = 3 * learning_rate
    return rate
##激活函数使用ReLU函数
def ReLU(x):
    if x >= 0:
        return x
    else :
        return x*1.0
##定义ReLU函数的导数
def ReLU_Derivative(x):
    if x >= 0:
        return 1.0
    else :
        return 1.0
##激活函数使用Sigmoid函数
def Sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
##定义Sigmoid函数的导数
def Sigmoid_Derivative(x):
    return x * (1-x)
##激活函数使用tanh函数
def tanh(x):
    return (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
##定义tanh函数的导数
def tanh_Derivative(x):
    return 1-x**2
##定义随机函数，用于随机数的产生
def random_maker(a,b):
    return random.uniform(a,b)
##定义神经网络类
class NeuralNetwork():
    ##参数初始化
    def _init_(self):
        ##各层细胞数量和输出层误差
        self.inputnum = 0
        self.hiddennum = 0
        self.outputnum = 0
        self.delta_o = 0.0
        ##用于存放各层神经元的值
        self.inputcells = []
        self.hiddencells = []
        self.outputcells = []
        ##用于存放各层权重和阈值
        self.inputweights = []
        self.outputweights = []
        self.inputweightsold = []
        self.outputweightsold = []
        self.hiddenthreshold = []
        self.outputthreshold = []
        self.hiddenthresholdold = []
        self.outputthresholdold = []
        ##用于存放各层误差
        self.hiddendeltas = []
        self.inputdeltas = []
    ##建立最初的网络
    #############################################################################
    # @作者    ：曹园
    # @函数简介: 网络建立并初始化
    # @输入参数: numin(输入层神经元个数，是一个整型数),numhidden(隐含层神经元个数，是一个整型数),numout(输出层神经元个数，是一个整型数)
    # @返回值  : 无
    # @其它: 其它
    #############################################################################
    def	setup(self,numin,numhidden,numout):
        ##输入各层细胞个数
        self.error_as = []
        self.inputnum = numin
        self.hiddennum = numhidden
        self.outputnum = numout
        ##初始化各层细胞的数值
        self.inputcells = np.zeros((1,self.inputnum))
        self.hiddencells = np.zeros((1,self.hiddennum))
        self.outputcells = np.zeros((1,self.outputnum))
        ##初始化各层的误差
        self.inputdeltas = np.zeros((1,self.inputnum))
        self.hiddendeltas = np.zeros((1,self.hiddennum))
        ##初始化各级权重、阈值、误差
        self.inputweights = 0.1*(2*np.random.rand(self.inputnum,self.hiddennum)-1)
        self.outputweights = 0.1*(2*np.random.rand(self.hiddennum,self.outputnum)-1)
        self.hiddenthreshold = 0.1*(2*np.random.rand(1,self.hiddennum)-1)
        self.outputthreshold = 0.1*(2*np.random.rand(1,self.outputnum)-1)
        a = 0.7**self.inputnum * math.sqrt(self.hiddennum)
        total = 0.0
        for i in range(self.inputnum):
            for h in range(self.hiddennum):
                total += self.inputweights[i][h]**2
        self.inputweights = a * self.inputweights / math.sqrt(total)
        total = 0.0
        for h in range(self.hiddennum):
            for o in range(self.outputnum):
                total += self.outputweights[h][0]**2
        self.outputweights = a * self.outputweights / math.sqrt(total)
        self.inputweightsold = self.inputweights.copy()
        self.outputweightsold = self.outputweights.copy()
        self.hiddenthresholdold = self.hiddenthreshold.copy()
        self.outputthresholdold = self.outputthreshold.copy()
    ##前向
    #############################################################################
    # @作者    ：曹园
    # @函数简介: 前向过程，输入样本得到网络的输出
    # @输入参数: input(输入样本，用于传递给前向过程)
    # @返回值  : self.outputcells[0][0](本次迭代的输出，是一个一维数组)
    # @其它: 其它
    #############################################################################
    def forward(self,input):
        ##激活输入层
        for i in range(self.inputnum):
            self.inputcells[0][i] = ReLU(input[i])
        ##激活隐藏层
        self.hiddencells = np.dot(self.inputcells,self.inputweights) + self.hiddenthreshold
        for h in range(self.hiddennum):
            self.hiddencells[0][h] = tanh(self.hiddencells[0][h])
        ##激活输出层
        self.outputcells = np.dot(self.hiddencells,self.outputweights) + self.outputthreshold
        for o in range(self.outputnum):
            self.outputcells[0][o] = ReLU(self.outputcells[0][o])
        return self.outputcells[0][0]
    ##错误率判断
    #############################################################################
    # @作者    ：曹园
    # @函数简介: 误差检测，计算本次迭代的误差和损失函数值并返回
    # @输入参数: input(输入样本，用于传递给前向过程)，desired_ouptputs(输入样本的期望值，是一个多维数组)
    # @返回值  : error(此次迭代的误差，是一个一维数组)，output(本次迭代的输出，是一个一维数组)
    # @其它: 其它
    #############################################################################
    def error_detection(self,input,desired_output):
        output = self.forward(input)
        error = 0.5 * (desired_output - output)**2
        return error,output
    ##反向误差传播
    #############################################################################
    # @作者    ：曹园
    # @函数简介: 误差反向传播过程，计算各层误差，并利用误差更新各层权重和阈值
    # @输入参数: input(输入样本，用于传递给前向过程)，desired_ouptputs(输入样本的期望值，是一个多维数组),learning_rate(学习率，是一个浮点数，取值为0~0.5),
    #           meomentum_rate(动量项，是一个浮点数，取值为0.0~0.9)
    # @返回值  : error(此次迭代的误差，是一个一维数组)
    # @其它: 其它
    #############################################################################
    def back_propagation(self,input,desired_output,learning_rate,meomentum_rate):
        error,output = self.error_detection(input,desired_output)
        ##计算输出层的误差
        self.delta_o = (desired_output - output) * ReLU_Derivative(output)
        ##计算隐含层的误差
        for h in range(self.hiddennum):
            self.hiddendeltas[0][h] = tanh_Derivative(self.hiddencells[0][h]) * self.delta_o * self.outputweights[h][0]
        ##输出层阈值的更新
        for o in range(self.outputnum):
            b = self.outputthreshold[0][o]
            self.outputthreshold[0][o] += learning_rate * self.delta_o + meomentum_rate * (self.outputthreshold[0][o] - self.outputthresholdold[0][o])
            self.outputthresholdold[0][o] = b
        ##隐含层权重和阈值的更新
        for h in range(self.hiddennum):
            a = self.outputweights[h][0]
            b = self.hiddenthreshold[0][h]
            self.outputweights[h][0] += learning_rate * self.delta_o * self.hiddencells[0][h] + meomentum_rate * (self.outputweights[h][0] - self.outputweightsold[h][0])
            self.hiddenthreshold[0][h] += learning_rate * self.hiddendeltas[0][h] + meomentum_rate * (self.hiddenthreshold[0][h] - self.hiddenthresholdold[0][h])
            self.outputweightsold[h][0] = a
            self.hiddenthresholdold[0][h] = b
        ##输入层权重的更新
        for i in range(self.inputnum):
            for h in range(self.hiddennum):
                a = self.inputweights[i][h]
                self.inputweights[i][h] += learning_rate * self.hiddendeltas[0][h] * self.inputcells[0][i] + meomentum_rate * (self.inputweights[i][h] - self.inputweightsold[i][h])
                self.inputweightsold[i][h] = a
        return error
    #############################################################################
    # @作者    ：曹园
    # @函数简介: 训练样本的训练，计算一次迭代的所有样本损失函数，每十次迭代输出当前的误差和损失函数值，当达到迭代次数或收敛时结束训练
    # @输入参数: inputs(训练样本，是一个多维数组)，desired_ouptputs(训练样本的期望值，是一个多维数组),learning_rate(学习率，是一个浮点数，取值为0~0.5),
    #           meomentum_rate(动量项，是一个浮点数，取值为0.0~0.9),limit(迭代次数，是一个整型数，取值为0-50000),correction(收敛误差，是一个浮点数，取值为0.001~0.003)
    # @返回值  : count(实际迭代次数，是一个整型数，≤limit),self.error_as(误差输出，是一个一维数组，用于绘制误差曲线)
    # @其它: 其它
    #############################################################################
    def train(self,inputs,desired_outputs,learning_rate,meomentum_rate,limit,correction):
        count = 0
        counta = 0
        rate1 = learning_rate
        ##开始迭代
        while(1):
            error = 0.0
            for i in range(inputs.shape[0]):
                for j in range(inputs.shape[1]):
                    desired_output = desired_outputs[i][j]
                    input = inputs[i][j]
                    error += self.back_propagation(input,desired_output,learning_rate,meomentum_rate)
            count += 1
            rate1 = rate(learning_rate,error)
            self.error_as.append(error)
            ##当损失函数小于给定的收敛误差，认为其已经收敛，计数
            if error < correction :
                counta += 1
            ##每十次迭代输出当前的误差和损失函数值
            if count % 10 == 0 :
                print (count,error,self.delta_o)
            ##当达到迭代次数或收敛时结束训练
            if (count>limit or counta > 1000):
                break
        return count,self.error_as
    #############################################################################
    # @作者    ：曹园
    # @函数简介: 检测样本的检测
    # @输入参数: inputs(检测样本，是一个多维数组)
    # @返回值  : Y_out[:](检测样本的输出，是一个一维数组)   
    # @其它: 其它
    #############################################################################
    def test(self,inputs):
        for i in range(inputs.shape[0]):
                for j in range(inputs.shape[1]):
                    input = inputs[i][j]
                    Y_out.append(self.forward(input))
        return Y_out[:]

if __name__ == '__main__':
    ##问题一的解，请将问题二的解下的代码注释
    nn = NeuralNetwork()
    ##设置神经网络的结构
    nn.setup(1,30,1)
    ##设置神经网络的参数
    count,error_out = nn.train(X_train1,Y_train1, 0.05, 0.5,2000,0.001)
    ##检测样本的测试
    nn.test(X_test1)
    aa = [i for i in range(count)]
    ##各多维数组维度变化
    X_train1 = X_train1.flatten() 
    Y_train1 = Y_train1.flatten() 
    X_test1 = X_test1.flatten() 
    Y_test1 = Y_test1.flatten() 
    X_train1 = X_train1.tolist()
    Y_train1 = Y_train1.tolist()
    X_test1 = X_test1.tolist()
    Y_test1 = Y_test1.tolist()
    error_out = np.array(error_out)
    error_aaa = error_out.sum()
    ##输出均方误差
    print(error_aaa/count)
    error_out = error_out.flatten()
    error_out = error_out.tolist()
    ##作图查看输出结果
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(X_train1,Y_train1,'r*')
    plt.plot(X_test1,Y_out,color="blue", linewidth=1.0, linestyle="-")
    plt.plot(X_test1,Y_test1,color="green", linewidth=1.0, linestyle="-")
    plt.subplot(2,1,2)
    plt.plot(aa,error_out,color="red", linewidth=1.0, linestyle="-")
    plt.show()
    ##问题二的解，请将问题一的解下的代码注释
    """ nn = NeuralNetwork()
    ##设置神经网络的结构
    nn.setup(2,50,1)
    ##设置神经网络的参数
    count,error_out = nn.train(X_train2,Y_train2, 0.0015, 0.8,10000,0.001)
    ##检测样本的测试
    nn.test(X_test2)
    aa = [i for i in range(count)]
    ##各多维数组维度变化
    X_test2 = X_test2.flatten()
    X_test2 = X_test2.tolist()
    X1 = [0.0 for i in range(21)]
    X2 = [0.0 for i in range(21)]
    for i in range(21):
        X1[i] = X_test2[2*i+1]
        X2[i] = X_test2[2*i+1]
    fig = plt.figure()
    ax = Axes3D(fig)
    X1 = np.array(X1)
    X2 = np.array(X2)
    X1,X2 = np.meshgrid(X1,X2)
    Y_out = np.array(Y_out).reshape(21,21)
    error_out = np.array(error_out)
    error_aaa = error_out.sum()
    ##输出均方误差
    print(error_aaa/count)
    error_out = error_out.flatten()
    error_out = error_out.tolist()
    ##作图查看输出结果
    plt.figure()
    ax.plot_surface(X1, X2, Y_out, rstride=1, cstride=1, cmap='rainbow') 
    plt.plot(aa,error_out,color="red", linewidth=1.0, linestyle="-")
    plt.show() """
