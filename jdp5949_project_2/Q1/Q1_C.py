import numpy as np
import math as m
from matplotlib import pyplot as plt


'''
approch:
1. data cleaning and formatting using old code which provided by TA last time
2. There are two list(array) x and y 
3. calculate theta for test data tx,ty for different k (1-10) and d(0-6) values
4. To calculate theta I used pseudoinverse : theta= x^-1 * y (where x is calculated from given sin(k*i*x) formula)
5. Using theta I calculated predicted Y using given formula:y=theta0+theta1*sin(i*k*x)+...
6. add all the predicted new Y point into "Prearr"
7. using predicted Y and original Y I calcualted mean sqaure error for each k and d value print them on terminal
'''

'''
Which ”function depth” would you consider the best prediction function and why? : > funuctional depth 6 is best in all k because with increment functional depth MSE is decresing.  

For which values of k and d do you get minimum error? :> k=7 d=6 MSE=4.431278223659305e-28 which is amlost 0

Compare the error results and try to determine for what “function depths” overfitting might be a problem. : error increase with k so higher k has overfitting problem. 
'''

def main():
    print('START Q1_C\n')
    # calcualte error using MSE
    def errorT(x,xt):
        temp=0
        temp=np.square(x-xt) #using numpy square of diffrence benween orignal and predicted Y
        return temp.mean()
    
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
                
                arr.append(m.sin(temp)**2)#generate sin ans for diffrent x,k,i value
            masterArr.append(arr)
        mArr=np.array(masterArr)#converted arr to numpy
        inX=np.linalg.pinv(mArr)#pseudoinverse for sin answer array
        t=np.dot(inX,y)#matrix multiplication 
        return t

    FilePath='../jdp5949_project_2/datasets/Q1_B_train.txt'#relative file path
    Dataset1=readFile(FilePath)
    x,y=[],[]#for further usage I have created new x,y
    for i in range(len(Dataset1)):
        x.append(float(Dataset1[i][0]))
    for i in range(len(Dataset1)):
        y.append(float(Dataset1[i][1]))

    #created tx,ty for test input and formatted into xt,yt list
    FilePathTest='../jdp5949_project_2/datasets/Q1_C_test.txt'
    TestSet=readFile(FilePathTest)
    xt,yt=[],[]
    for i in range(len(TestSet)):
        xt.append(float(TestSet[i][0]))
    for i in range(len(TestSet)):
        yt.append(float(TestSet[i][1]))
    TNumX=np.array(xt)#making numpy array from list
    TNumY=np.array(yt)
    NormalTestDataX=TNumX
    NormalTestDataY=TNumY
    masterAray=[]
    T=[]
    for k in range(1,11):#interate k from 1 to 10
        for d in range(0,7):#interate d from 0 to 6
            theta=Theta(k,d,NormalTestDataX,NormalTestDataY)#get theta for every k,d values
            PreArr=[]
            for i in range(len(NormalTestDataY)):#iterate all value to calculate new precited y
                count=0
                ans=0
                for j in theta:
                    if count==0:
                        ans+=j
                    else:
                        ans+=j*((m.sin(k*count*NormalTestDataX[i]))**2)#sum of whole given equation
                    count+=1
                PreArr.append(ans)
            # print(k,d,PreArr)
            T.append([k,d,errorT(NormalTestDataY,PreArr)])#storing error value with k and d for my code debugging understanding
            masterAray.append([errorT(NormalTestDataY,PreArr)])#calculate and storing all errors into masterAray
    for i in range(len(T)):
        print("K =",T[i][0]," d =",T[i][1]," MSE =",T[i][2])#print errors 
    print('END Q1_C\n')


if __name__ == "__main__":
    main()
