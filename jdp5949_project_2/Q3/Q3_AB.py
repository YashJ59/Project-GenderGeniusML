import numpy as np
import math as m
import matplotlib.pyplot as plt
import random as r

'''
Approch for Q3-AB:
1. data cleaning and formatted using TA's code
2. generate perameter matrix using random values between 0-1
3. Calculating sigmoid for genrated theta and given data
4. if sigmoid values is 0.5 or more and gender class is M 
5. calculate error using prediction class by using sigmoid
6. using greadient decent we calcualte accuracy agaom and again for 1000 times 
7. return last thetas from 1000 interation
8. using theta genrate new feture matrix
9. by usnig theta and feture matrix calculate predicct the class for overall accuracy 
10. using data points and predicted data points for gender label create hyper plan   
'''

def main():
	print('START Q3_AB\n')
	'''
	Start writing your code here
	'''
	lr=0.01#leaning rate
	apNum=2000
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

	def sigmoid(t, inp):#sigmoid function
		temp,temp1,temp2=0,0,0
		temp = np.dot(t, inp)
		temp2=(1 + m.exp(-temp))
		temp1=1 / temp2#calculation of sigmoid function using sigmoid formula
		return temp1

	large_120_data = '../jdp5949_project_2/datasets/Q3_data.txt'#data 
	#data formatting 
	t0,t1,t2,t3=[],[],[],[]
	set2 = readFile(large_120_data)
	h,w,a,g=[],[],[],[]
	pm=[]
	p=r.randrange(0,30)
	for i in range(len(set2)):
		p=r.randrange(0,30)
		h.append(float(set2[i][0]))
		w.append(float(set2[i][1]))
		a.append(float(set2[i][2]))
		if set2[i][3]=='M':
			g.append('M')
		else:
			g.append('W')
		t0.append(r.random())
		t1.append(r.random())
		t2.append(r.random())
		t3.append(r.random())
	for i in range(len(set2)):
		pm.append([t0[i],t1[i],t2[i],t3[i]])
	tp = np.array([h, w, a])#created list with has [height,weight,age] 
	tdata=np.transpose(tp)#did traspose to make it less complex tof uture usage
	#finish data formatting 
	t = np.random.uniform(low=0, high=1, size=3)#generated random theta for perameter matrxi

	def predict_logistic_algoritham(ti,t):#logistric regression prediction function which return final calculated theta
    #whole function devided into saveral samller function to make code less complex
		def supportAlgo(ti,t):#support function calculate greadint decent for 1000 time to see improvemnt in acuracy 
			err=0
			c=0
			for i in range(apNum):
				inx = np.random.randint(0, 3)
				yt=sigmoid(t,ti[inx])
				yh = sigmoid(t, ti[inx])#calculated sigmoid for given train data and theta
				if genArr[inx] =="M":# saperating gender calculation 
					y=1
				else:
					y=0
				tmp=0.5
				if yh <=tmp:#if sigmoid ans is not in within range then error is incremented  
					if y==1:
						err=err+1
				if yh>tmp:
					if y==0:
						err+=1
				ydiff=y-yh
				t -= lr * ydiff* ti[inx]# greadient decent to find best accuracy possible doing minus from peramener matrix to learning rate*alpha*feture matrix
				ifVar=(i+1)
				ifVar=ifVar%100# to minimize total intertaions 
				
				if ifVar== 0:
					c+=1
					print("Accuracy: ",p(err,i),"For Interation=",c)# printing accuracy for each ieration
					v=err/100
					if v <= lr:
						print("Learning rate is less than ",lr,"So loop break")# if accuracy goes below learning rate it will break loop 
						break
					err = 0
			return t
		def p(err,i):#calculating erro
			return (err/100)
		err ,te= 0,0
		def perameterMatPre(pm):# generate perameter matrix for calcualtion
			PreArr=[]
			for i in range(len(set2)):
				y=0
				y=pm[i][0]+(pm[i][1]*h[i])+(pm[i][2]*w[i])+(pm[i][3]*a[i])# formula t0+t1*h+...
				PreArr.append(y)
			return PreArr
		def sigmoidArrAndError():# calculate errors from sigmoid ans
			sigmoidAns=[]
			for i in range(len(set2)):
				sigmoidAns.append(([i]))
			count=0
			for i in range(len(set2)):
				if sigmoidAns[i][0]<0.5 and g[i]==1:
					count+=1
				elif sigmoidAns[i][0]>0.5 and g[i]==0:
					count+=1
			e=(len(set2)-count)/len(set2)#calcualte error
			return()
		return supportAlgo(ti,t)#return calculated final theta from gredient decent 
	t = np.random.uniform(low=0, high=1, size=3)
	genArr=np.array(g)#gender arra converted to numpy arrary
	train=predict_logistic_algoritham(tdata,t)#getting theta from losgictric function
	
	PreArr=[]

	for i in range(len(set2)):
		y=0
		y=(train[0]*h[i])+(train[1]*w[i])+(train[2]*a[i])
		PreArr.append(y)#using y formula calculate predicted class
	newPreArr=[]
	for i in range(len(set2)):
		newPoint=0
		newPoint=sigmoid(t,tdata[i])
		if newPoint>0.5 and g[i]=='M':
			newPreArr.append(1)
		else:
			newPreArr.append(0)
	xm,ym,zm,xw,yw,zw=[],[],[],[],[],[]
	for i in range(len(w)):#saperating data point based on gender
		if g[i]=='M':
			xm.append(h[i])
			ym.append(w[i])
			zm.append(a[i])
		else:
			xw.append(h[i])
			yw.append(w[i])
			zw.append(a[i])		
	ax = plt.axes(projection ='3d')
	ax.plot3D(xm,ym,zm,'.g')#men points 
	ax.plot3D(xw,yw,zw,'.y')#women points	

	plt.show()
	
	print('END Q3_AB\n')


if __name__ == "__main__":
    main()
    