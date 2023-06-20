import numpy as np
import random as r
import math as m

'''
IMporatnat Note: Due to data complextiy and huge calculation involved This code take some time to run. like 10 sec to 45 sec depending on your pc spacification
'''

'''
Approch for Q3-AB:
>for leave-out data formatting I used my old code from project one
> then repated all the steps from quetion 3AB and find accuracy
1. data cleaning and formatted using TA's code
2. generate perameter matrix using random values between 0-1
3. Calculating sigmoid for genrated theta and given data
4. if sigmoid values is 0.5 or more and gender class is M 
5. calculate error using prediction class by using sigmoid
6. using greadient decent we calcualte accuracy agaom and again for 1000 times 
7. return last thetas from 1000 interation
8. using theta genrate new feture matrix
9. by usnig theta and feture matrix calculate predicct the class for overall accuracy 
'''
'''
compare the results with the ones for KNN and Naïve Bayes Discuss what differences exist and why one method might outperform the others for this problem.
> for leaveout knn and Naïve Bayes and logistic : knn and naive byayes accuracy is around 60-70% however logistric deeps down to around 50-60% which is significant changes. 
> it is because of naturenof algoritham, knn and naive bayses have same algoritham  
'''
def main():
	print('START Q3_C\n')
	'''
	Start writing your code here
	'''
	lr=0.01
	apNum=1000
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
	tp = np.array([h, w, a])
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
	masterTrainingDataA=[]
	for i in range(len(w)):
		p=r.randrange(1,30)
		testa=[]
		testx=[]
		testy=[]
		trainx=[]
		trainy=[]
		traina=[]
		traing=[]
		ans=[]
		for j in range(len(w)):
			if i==j:
				testx=h[j]
				testy=w[j]
				testa=a[j]
			else:
				trainx.append(h[j])
				trainy.append(w[j])
				traina.append(a[j])
				traing.append(g[j])
		masterTrainingDataX.append(trainx)
		masterTrainingDataY.append(trainy)
		masterTrainingDataA.append(traina)
		masterans.append(ans)
		masterGen.append(traing)
	def predict_logistic_algoritham(ti,t):
		def supportAlgo(ti,t):
			err=0
			for i in range(apNum):
				inx = np.random.randint(0, 3)
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
				ifVar=ifVar%50
				if ifVar== 0:
					v=err/1000
					if v <= lr:
						break
					err = 0
			return t
		def p(err,i):
			return sigmoidArrAndError()[0],err/50, "On i=",i + 1
		err ,te= 0,0
		def perameterMatPre(pm):
			PreArr=[]
			for i in range(len(set2)):
				y=0
				y=pm[i][0]+(pm[i][1]*h[i])+(pm[i][2]*w[i])+(pm[i][3]*a[i])
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
			data.append([masterTrainingDataX[j][i],masterTrainingDataY[j][i],masterTrainingDataA[j][i]])
		t = np.random.uniform(low=0, high=1, size=3)
		masterData.append(data)
	t = np.random.uniform(low=0, high=1, size=3)
	genArr=np.array(g)
	for i in range(len(w)):
		train=[]
		train=predict_logistic_algoritham(masterData[i],t)
		masterTrain.append(train)
	samples=tdata

	
	for i in range(len(samples)):
		y_hat = sigmoid(masterTrain[i], samples[i])
		if y_hat>=0.5 and g[i]=='M' or y_hat<0.5 and g[i]=='W':
			p+=1
		else:
			p
		# p =p+ 1 if y_hat >= 0.5 and g[i]=='M' or y_hat<0.5 and g[i]=='W' else p
	print("Accuracy : ",p/len(w))
 
	print('END Q3_C\n')


if __name__ == "__main__":
    main()
    
