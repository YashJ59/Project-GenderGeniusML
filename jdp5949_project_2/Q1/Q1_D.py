import numpy as np
import math as m
from matplotlib import pyplot as plt

'''
approch for Q1-AB:
1. data cleaning and formatting using old code which provided by TA last time
2. There are two list(array) x and y 
3. calculate theta for different k (1-10) and d(0-6) values
4. To calculate theta I used pseudoinverse : theta= x^-1 * y (where x is calculated from given sin(k*i*x) formula)
5. Using theta I calculated predicted Y using given formula:y=theta0+theta1*sin(i*k*x)+...
6. add all the predicted new Y point into "Prearr"
7. Using new data point created graph for each k and d value
'''
'''
approch for Q1-C:
1. data cleaning and formatting using old code which provided by TA last time
2. There are two list(array) x and y 
3. calculate theta for test data tx,ty for different k (1-10) and d(0-6) values
4. To calculate theta I used pseudoinverse : theta= x^-1 * y (where x is calculated from given sin(k*i*x) formula)
5. Using theta I calculated predicted Y using given formula:y=theta0+theta1*sin(i*k*x)+...
6. add all the predicted new Y point into "Prearr"
7. using predicted Y and original Y I calcualted mean sqaure error for each k and d value print them on terminal
'''
'''
What differences do you see and why might they occur? : > First of all MSE is increses in all k and d because of less data point avaiable for prediction. 
> Graph is making no sense now because lesser data generate wrong prediction and wrong prediction leads to wrong graph
'''
def main():
    print('START Q1_D\n')
    
    def errorT(x,xt):
        temp=0
        temp=np.square(x-xt)
        return temp.mean()
    
    
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

    def Theta(k,d,x,y):
        
        masterArr=[]
        for i in range(len(x)):
            arr=[]
            arr.append(1)
            for j in range(1,d+1):
                temp=j*k*x[i]
                
                arr.append(m.sin(temp)**2)
            masterArr.append(arr)
        mArr=np.array(masterArr)
        inX=np.linalg.pinv(mArr)
        t=np.dot(inX,y)
        return t

    FilePath='../jdp5949_project_2/datasets/Q1_B_train.txt'
    Dataset1=readFile(FilePath)
    x,y=[],[]
    for i in range(len(Dataset1)):
        x.append(float(Dataset1[i][0]))
    for i in range(len(Dataset1)):
        y.append(float(Dataset1[i][1]))
    NumX=np.array(x[:20])#first 20 data points
    NumY=np.array(y[:20])#first 20 data points
    FilePathTest='../jdp5949_project_2/datasets/Q1_C_test.txt'
    TestSet=readFile(FilePathTest)
    xt,yt=[],[]
    for i in range(len(TestSet)):
        xt.append(float(TestSet[i][0]))
    for i in range(len(TestSet)):
        yt.append(float(TestSet[i][1]))
    masterAray=[]
    T=[]

    masterArr=[]
    #for prediction points and error calculation 
    for k in range(1,11):#interate k from 1 to 10
        for d in range(0,7):#interate d from 0 to 6
            
            theta=Theta(k,d,NumX,NumY)#get theta for every k,d values
            PreArr=[]
            for i in range(len(NumY)):#iterate all value to calculate new precited y
                count=0
                ans=0
                for j in theta:
                    if count==0:
                        ans+=j
                    else:
                        ans+=j*((m.sin(k*count*NumX[i]))**2)#sum of whole given equation
                    count+=1
                PreArr.append(ans)#stroring ans
            T.append([k,d,errorT(NumY,PreArr)])#error calcualte
            masterAray.append([errorT(NumY,PreArr)])                
            masterArr.append(PreArr)#storing all ans
    for i in range(0,60,6):# generate graph for each k and d values
        plt.title("Training Data Size=128")
        plt.plot(NumX,NumY,'.')
        plt.plot(NumX,masterArr[i],'.y',label='d=0')
        plt.plot(NumX,masterArr[i+1],'.b',label='d=1')
        plt.plot(NumX,masterArr[i+2],'.g',label='d=2')
        plt.plot(NumX,masterArr[i+3],'.r',label='d=3')
        plt.plot(NumX,masterArr[i+4],'.m',label='d=4')
        plt.plot(NumX,masterArr[i+5],'.c',label='d=5')
        plt.legend()
        plt.show()
    for i in range(len(T)):
        print("K =",T[i][0]," d =",T[i][1]," MSE =",T[i][2])#print error
    print('END Q1_D\n')


if __name__ == "__main__":
    main()
