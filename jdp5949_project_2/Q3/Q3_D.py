import numpy as np
import math as m
import matplotlib.pyplot as plt
import random as r

'''
deleted age list
and repete same steps from Q3-AB
'''

def main():
	print('START Q3_D\n')
	'''
	Start writing your code here
	'''
	lr=0.01
	apNum=100
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
	large_120_data = '../jdp5949_project_2/datasets/Q3_data.txt'

	t0,t1,t2,t3=[],[],[],[]
	set2 = readFile(large_120_data)
	h,w,a,g=[],[],[],[]
	pm=[]
	for i in range(len(set2)):
		h.append(float(set2[i][0]))
		w.append(float(set2[i][1]))#removed age
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
	tp = np.array([h, w])
	tdata=np.transpose(tp)
	t = np.random.uniform(low=0, high=1, size=3)
	def sigmoid(t, inp):
		temp,temp1,temp2=0,0,0
		temp = np.dot(t, inp)
		temp2=(1 + m.exp(-temp))
		temp1=1 / temp2
		return temp1
	masterans=[]
	masterGen=[]
	masterTrainingDataX=[]
	masterTrainingDataY=[]
	for i in range(len(w)):
		p=r.randrange(1,20)
		trainx=[]
		trainy=[]
		traing=[]
		ans=[]
		for j in range(len(w)):
			if i==j:
				testx=h[j]
				testy=w[j]
			else:
				trainx.append(h[j])
				trainy.append(w[j])
				traing.append(g[j])
		masterTrainingDataX.append(trainx)
		masterTrainingDataY.append(trainy)
		masterGen.append(traing)
	def predict_logistic_algoritham(ti,t):
		def supportAlgo(ti,t):
			err=0
			for i in range(apNum):
				inx = np.random.randint(0, 2)
				yt=sigmoid(t,ti[inx])
				yh = sigmoid(t, ti[inx])
				if genArr[inx] =="M":
					y=1
				else:
					y=0
				tmp=0.5
				if yh <=tmp:
					if y==1:
						err=err+1
				if yh>tmp:
					if y==0:
						err+=1
				ydiff=y-yh
				t=t*ydiff
				tIndex=t[inx]
				t=t*tIndex
				ifVar=(i+1)
				ifVar=ifVar%100
				if ifVar== 0:
					v=err/1000
					if v <= lr:
						break
					err = 0
			return t
		def p(err,i):
			return sigmoidArrAndError()[0],err/100, "On i=",i + 1
		err ,te= 0,0
		def perameterMatPre(pm):
			PreArr=[]
			for i in range(len(set2)):
				y=0
				y=pm[i][0]+(pm[i][1]*h[i])+(pm[i][2]*w[i])
				PreArr.append(y)
			return PreArr
		def sigmoidArrAndError():
			sigmoidAns=[]
			for i in range(len(set2)):
				sigmoidAns.append(([i]))
			count=0
			for i in range(len(set2)):
				if sigmoidAns[i][0]<0.5 and g[i]==1:
					count+=1
				elif sigmoidAns[i][0]>0.5 and g[i]==0:
					count+=1
			e=(len(set2)-count)/len(set2)
			return("Error",e)
		return supportAlgo(ti,t)
	masterData=[]
	masterTrain=[]
	for j in range(len(w)):
		data=[]
		for i in range(len(w)-1):
			data.append([masterTrainingDataX[j][i],masterTrainingDataY[j][i]])
		t = np.random.uniform(low=0, high=1, size=2)
		masterData.append(data)
	t = np.random.uniform(low=0, high=1, size=2)
	genArr=np.array(g)
	for i in range(len(w)):
		train=[]
		train=predict_logistic_algoritham(masterData[i],t)
		masterTrain.append(train)
	samples=tdata

	
	for i in range(len(samples)):
		y_hat = sigmoid(masterTrain[i], samples[i])
		p =p+ 1 if y_hat >= 0.5 and g[i]=='M' or y_hat<0.5 and g[i]=='W' else p
	print("Accuracy : ",p/len(w))
	PreArr=[]


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

	(x, y) = np.meshgrid(np.arange(min(h),max(h),0.02), np.arange(min(a), max(a), 0.2))#drawing hyper plan by using perameter matrx and put perameter matrix into Predicted list 
	z = x+y#to making 2d into z diamentation

	ax.plot_surface(x, y, z)#ploting hyper plan
	plt.show()
	print('END Q3_D\n')


if __name__ == "__main__":
    main()
    
