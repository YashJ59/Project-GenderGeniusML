import numpy as np
import math as m
from matplotlib import pyplot as plt

#create function which add 1000 new data point.




'''
approch:
1. data cleaning and formatting using old code which provided by TA last time
2. There are two list(array) x and y 
3. calculate theta for different k (1-10) and d(0-6) values
4. To calculate theta I used pseudoinverse : theta= x^-1 * y (where x is calculated from given sin(k*i*x) formula)
5. Using theta I calculated predicted Y using given formula:y=theta0+theta1*sin(i*k*x)+...
6. add all the predicted new Y point into "Prearr"
7. Using new data point created graph for each k and d value
'''
def main():
    print('START Q1_AB\n')
    # Data cleaning and formatting start
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
    # Data cleaning and formatting start
    def Theta(k,d,x,y):#calculating theta for any k and d value
        
        masterArr=[]
        for i in range(len(x)):
            arr=[]
            arr.append(1)#added 1 into feture matrix at start because of theta0*1
            for j in range(1,d+1):
                temp=j*k*x[i]#sin input
                
                arr.append(m.sin(temp)**2)#generate sin's answer for diffrent x,k,i value
            masterArr.append(arr)
        mArr=np.array(masterArr)#converted arr to numpy
        inX=np.linalg.pinv(mArr)#pseudoinverse for sin answer array
        t=np.dot(inX,y)#matrix multiplication pseudoinverse with y array
        return t#return the answer

    FilePath='../jdp5949_project_2/datasets/Q1_B_train.txt'#relative file path
    Dataset1=readFile(FilePath)
    x,y=[],[]#for further usage I have created new x,y
    for i in range(len(Dataset1)):
        x.append(float(Dataset1[i][0]))
    for i in range(len(Dataset1)):
        y.append(float(Dataset1[i][1]))
    NumX=np.array(x)
    NumY=np.array(y)
    #created tx,ty for test input just for future use
    FilePathTest='../jdp5949_project_2/datasets/Q1_C_test.txt'#test file
    TestSet=readFile(FilePathTest)
    xt,yt=[],[]
    for i in range(len(TestSet)):
        xt.append(float(TestSet[i][0]))
    for i in range(len(TestSet)):
        yt.append(float(TestSet[i][1]))

    
    masterArr=[]

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
            masterArr.append(PreArr)#storing all ans
            


    for i in range(0,70,7):# generate graph for each k and d values
        plt.title("Training Data Size=128")
        plt.plot(NumX,NumY,'.')
        plt.plot(NumX,masterArr[i],'.y',label='d=0')#d=0
        plt.plot(NumX,masterArr[i+1],'.b',label='d=1')#d=1
        plt.plot(NumX,masterArr[i+2],'.g',label='d=2')#d=2
        plt.plot(NumX,masterArr[i+3],'.r',label='d=3')#d=3
        plt.plot(NumX,masterArr[i+4],'.m',label='d=4')#d=4
        plt.plot(NumX,masterArr[i+5],'.c',label='d=5')#d=5
        plt.plot(NumX,masterArr[i+5],'.c',label='d=6')#d=6
        plt.legend()
        plt.show()
    
    print('END Q1_AB\n')


if __name__ == "__main__":
    main()

