import numpy as np
import math as m
from matplotlib import pyplot as plt

''' 
Approach for Q2-C:
1. Thera are 3 function for data cleaninig and formatt8ing which was provided by TA in last project. I reused that code for make data clean 
2. created 2 list x and y which contain all data points
3. There is weight function which gives the weight of given array/list by using given  formula, all weight to one list and weight function return list of all weight of all data point
4. To calculate theta, I got instruction from alman,alok , multiply weights with x and y list then did pseudoinverse of x and then matrix multiplication with y. 
5. return thetas, which are diffent for each x and y values then using those theta calculated predicted y value. 
6. using predicted y and original y , I calculated error
'''
'''
    How does the performance compare to the one for the results from Question 1 c) ? : > due to the weight MSE is increase significantaly compare to 1C. 
    > some data point has higher weight so high chances for making error if that data point prediced wrong 
'''

def main():
    print('START Q2_C\n')
    def w(x,tx):#weight function to calculate weight for given data point
        masterWeight=[]
        for i in range(len(tx)):
            weight=[]
            for j in range(len(x)):
                w=0
                w=m.exp(-((x[j]-tx[i])**2)/(2*0.204*0.204))#calculated weighted given formula
                weight.append(w)
            masterWeight.append(weight)
        return masterWeight

    def MSE(x, xt):#calculate MSE error
        temp=0
        temp=np.square(x-xt) #using numpy square of diffrence benween orignal and predicted Y
        return temp.mean()#calculate mean of all squarae diffretionated points
    def clean_data(line):
        return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')

    def fetch_data(filename):
        with open(filename, 'r') as f:
            input_data = f.readlines()
            clean_input = list(map(clean_data, input_data))
            f.close()
        return clean_input


    def readFile(dataset_path):
        input_data = fetch_data(dataset_path)
        input_np = np.array(input_data)
        return input_np




    def Theta(w,x,y):#calculate theta using weight and x,y values
        masterX,masterY=[],[]
        for j in range(len(w)):
            xt,yt=[],[]
            for i in range(len(x)):
                temp,temp2=0,0
                temp=w[j][i]*x[i]#normal multiply weight with x list
                temp2=w[j][i]*y[i]#normal multiply weight with y list
                xt.append([1,temp])#added 1 for theta0
                yt.append(temp2)
            masterX.append(xt)
            masterY.append(yt)
        masterT=[]
        for j in range(len(w)):
            txt,tyt=[],[]
            inX=[]
            inY=[]
            t=[]
            inX=np.linalg.pinv(np.mat(masterX[j]))# calculate pseudoinverse for x values 
            inY=np.mat(masterY[j])#making y list to matrix 128*1
            t=np.dot(inX,inY.T)#matrix multiplication pseudoinverse with y matrix
            masterT.append(t)#added ans to list
        ans=np.array(masterT)#converted array from list
        return ans#return ans

    FilePath='../jdp5949_project_2/datasets/Q1_B_train.txt'#file path for train
    Dataset1=readFile(FilePath)
    x,y=[],[]
    for i in range(len(Dataset1)):
        x.append(float(Dataset1[i][0]))
    for i in range(len(Dataset1)):
        y.append(float(Dataset1[i][1]))
    NumX=np.array(x)
    NumY=np.array(y)
    
    FilePathTest='../jdp5949_project_2/datasets/Q1_C_test.txt'#file path for test
    TestSet=readFile(FilePathTest)
    xt,yt=[],[]
    for i in range(len(TestSet)):
        xt.append(float(TestSet[i][0]))
    for i in range(len(TestSet)):
        yt.append(float(TestSet[i][1]))
    TNumX=np.array(xt)
    TNumY=np.array(yt)
    PreArr=[]
    l=w(xt,xt)#calculate weight for test values
    theta=Theta(l,TNumX,TNumY)#calculate theta for test data points
    for i in range(len(xt)):
        t0=theta[i][0]#theta0
        t1=theta[i][1]#theta1
        PreArr.append(t0+t1*xt[i])#calculate new predicted y=theta0+theta1*x
    print("Data point",len(x),", MSE=",MSE(np.array(xt),np.array(PreArr)))#fine error between original data point and predicted points

    print('END Q2_C\n')


if __name__ == "__main__":
    main()

