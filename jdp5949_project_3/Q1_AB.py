import numpy as np
import math as m
import random as r

def main(pp):
    print('START Q1_AB\n')
    def split(arr):#data split function
        ans=[]
        for i in range(1,len(arr)):
            ans.append([arr[:i],arr[i:]])
        return ans    
    def SystemEntropy(arr):
        ans=[]
        ans.append(0)
        for i in range(1,len(arr)-1):
            temp=0
            temp1=0
            # print((arr[i][1][0]))
            temp=-(sum(arr[i][0])/len(arr))*(((arr[i][0][0]/sum(arr[i][0]))*m.log2((arr[i][0][0]/sum(arr[i][0]))))+((arr[i][0][1]/sum(arr[i][0]))*m.log2(arr[i][0][1]/sum(arr[i][0]))))
            temp1=-(sum(arr[i][1])/len(arr))*(((arr[i][1][0]/sum(arr[i][1]))*m.log2((arr[i][1][0]/sum(arr[i][1]))))+((arr[i][1][1]/sum(arr[i][1]))*m.log2(arr[i][1][1]/sum(arr[i][1]))))
            ans.append(temp+temp1)
        ans.append(0)
        return ans   
    def CalculateThresholdValues(arr):
        ans=[]
        for i in range(len(arr)-1):
            ans.append((arr[i]+arr[i+1])/2)
        return ans

    def entropy(arr):
        ans=[]
        for i in range(len(arr)):
            temp=0
            temp=-((((len(arr[i][0]))/(len(arr)))*m.log2(((len(arr[i][0]))/(len(arr)))))+(((len(arr[i][1]))/(len(arr)))*m.log2(((len(arr[i][1]))/(len(arr))))))
            ans.append(temp)
        return ans


    def InfoGain(earr,sysearr):
        ans=[]
        for i in range(len(earr)):
            ans.append(abs(sysearr[i]-earr[i]))
        return ans

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

    large_50_data = '../jdp5949_project_3/datasets/Q1_train.txt'#data 
    test_70_data='../jdp5949_project_3/datasets/Q1_test.txt'
    def testData(test_70_data):
        t0,t1,t2,t3=[],[],[],[]
        set2 = readFile(test_70_data)
        h,w,a,g=[],[],[],[]
        pm=[]
        for i in range(len(set2)):
            h.append(float(set2[i][0]))
            w.append(float(set2[i][1]))
            a.append(float(set2[i][2]))
            if set2[i][3]=='M':
                g.append(1)
            else:
                g.append(0)
        return h,w,a,g
    testList=np.array(testData((test_70_data))).T
    # print(testList)
    #data formatting 
    t0,t1,t2,t3=[],[],[],[]
    set2 = readFile(large_50_data)
    h,w,a,g=[],[],[],[]
    pm=[]
    for i in range(len(set2)):
        h.append(float(set2[i][0]))
        w.append(float(set2[i][1]))
        a.append(float(set2[i][2]))
        if set2[i][3]=='M':
            g.append(1)
        else:
            g.append(0)
    h=sorted(h)
    w=sorted(w)
    a=sorted(a)
    dataset_50_train = np.array([h, w, a,g]).T
    heightSplit=split(h)# all possible split data for height - 49 combination
    weightSplit=split(w)# all possible split data for weight- 49 combination
    ageSplit=split(a)# all possible split data for age- 49 combination
    GenSplit=split(g)
    GsplitCount=[]
    for i in range(len(g)-1):#counting men women for each data split combination [[men,women for left split][men,women for right split]]
        GsplitCount.append([[len((GenSplit[i][0]))-sum(GenSplit[i][0]),len((GenSplit[i][0]))-(len((GenSplit[i][0]))-sum(GenSplit[i][0]))],[len((GenSplit[i][1]))-sum(GenSplit[i][1]),len((GenSplit[i][1]))-(len((GenSplit[i][1]))-sum(GenSplit[i][1]))]])
    
    

    
    Tweight=CalculateThresholdValues(w) #calculated all possisble threshold values for weight - 49 values
    Theight=CalculateThresholdValues(h)#calculated all posisble threshold values for height - 49 values
    Tage=CalculateThresholdValues(a)#calculated all posisble threshold values for age - 49 values
    # print(Theight)
    Eweight=entropy(weightSplit)
    Eheight=entropy(heightSplit)
    Eage=entropy(ageSplit)
    sysEnt=SystemEntropy(GsplitCount)
    # print((Eweight))
    # print((Eheight))
    # print((Eage))
    InfoGainWeight=InfoGain(Eweight,sysEnt)
    InfoGainHeight=InfoGain(Eweight,sysEnt)
    InfoGainAge=InfoGain(Eage,sysEnt)
    # print(InfoGainWeight)
    allT=[]
    for i in range(len(Tweight)):
        allT.append(Tweight[i])
        allT.append(Theight[i])
        allT.append(Tage[i])
    allE=[]
    for i in range(len(Eweight)):
        allE.append(Eweight[i])
        allE.append(Eheight[i])
        allE.append(Eage[i])
    allInfoGain=[]
    for i in range(len(InfoGainHeight)):
        allInfoGain.append(InfoGainWeight[i])
        allInfoGain.append(InfoGainAge[i])
        allInfoGain.append(InfoGainHeight[i])
    def findbestpera(alle,allt,w):#get best perameter from all of the 147 values of threshold and entropy
        l=[]
        l.append([allt[alle.index(min(alle))],min(alle),len(w),[alle.index(min(alle)),len(w)-alle.index(min(alle))]])#memorize everything based on all threshold,entropy,sample,M,W into one list
        return l
    # print(findbestpera(allE,allT,w))
    class Tree:
        def __init__(self, data):
            self.data = data
            self.leftChild = None
            self.rightChild = None
    
    n1=Tree(findbestpera(allE,allT,w))

    def reccurantFun(newW,newH,newA):
        prevdepth=1
        something=[]
        h=sorted(h)
        w=sorted(w)
        a=sorted(a)
        dataset_50_train = np.array([h, w, a]).T
        heightSplit=split(h)# all possible split data for height - 49 combination
        weightSplit=split(w)# all possible split data for weight- 49 combination
        ageSplit=split(a)# all possible split data for age- 49 combination
        GenSplit=split(g)
        GsplitCount=[]
        for i in range(len(g)-1):#counting men women for each data split combination [[men,women for left split][men,women for right split]]
            GsplitCount.append([[len((GenSplit[i][0]))-sum(GenSplit[i][0]),len((GenSplit[i][0]))-(len((GenSplit[i][0]))-sum(GenSplit[i][0]))],[len((GenSplit[i][1]))-sum(GenSplit[i][1]),len((GenSplit[i][1]))-(len((GenSplit[i][1]))-sum(GenSplit[i][1]))]])       
        Tweight=CalculateThresholdValues(w) #calculated all possisble threshold values for weight - 49 values
        Theight=CalculateThresholdValues(h)#calculated all posisble threshold values for height - 49 values
        Tage=CalculateThresholdValues(a)#calculated all posisble threshold values for age - 49 values
        # print(Theight)
        Eweight=entropy(weightSplit)
        Eheight=entropy(heightSplit)
        Eage=entropy(ageSplit)
        sysEnt=SystemEntropy(GsplitCount)
        # print((Eweight))
        # print((Eheight))
        # print((Eage))
        InfoGainWeight=InfoGain(Eweight,sysEnt)
        InfoGainHeight=InfoGain(Eweight,sysEnt)
        InfoGainAge=InfoGain(Eage,sysEnt)
        # print(InfoGainWeight)
        allT=[]
        for i in range(len(Tweight)):
            allT.append(Tweight[i])
            allT.append(Theight[i])
            allT.append(Tage[i])
        allE=[]
        for i in range(len(Eweight)):
            allE.append(Eweight[i])
            allE.append(Eheight[i])
            allE.append(Eage[i])
        allInfoGain=[]
        for i in range(len(InfoGainHeight)):
            allInfoGain.append(InfoGainWeight[i])
            allInfoGain.append(InfoGainAge[i])
            allInfoGain.append(InfoGainHeight[i])
        if Tree.data!=None and entropy==0:#pure leave condition
            return [0,len(newW)]
        else:
            if Tree.leftChild == None and findbestpera(allE,allT,newW)[3][0]:
                Tree.leftChild=reccurantFun(allE,allT,newW)[3][0]
            if Tree.rightChild == None and findbestpera(allE,allT,newW)[3][1]:
                Tree.rightchild=reccurantFun(allE,allT,newW)[3][1]
            if depth==prevdepth:
                pass
            else:
                depth+=1
                something.append(depth)
            return Tree
    def Supportcalaccdepth():
        for i in range(5):
            CalaccDepth(i+1,r.choice(allE),r.choice(allE))
            
    def calAccuracyTest(Tree,tl):
        ans=[]
        c=0
        while tl:
            if Tree.rightChild != None and tl[0] >=findbestpera(allE,allT,newW)[0] or tl[1] >=findbestpera(allE,allT,newW)[0] or tl[2]>=findbestpera(allE,allT,newW)[0]:
                if findbestpera(allE,allT,newW)[0]==0:
                    ans.append(max(findbestpera(allE,allT,newW)[3]))   
        for i in range(len(tl)):
            if tl[1][i][4]==ans[i]:
                c+=1
        return (c/len(tl))
    def calAccuracyTrain(Tree,dataset_50_train):
        ans=[]
        c=0
        while dataset_50_train:
            if Tree.rightChild != None and dataset_50_train[0] >=findbestpera(allE,allT,newW)[0] or dataset_50_train[1] >=findbestpera(allE,allT,newW)[0] or dataset_50_train[2]>=findbestpera(allE,allT,newW)[0]:
                if findbestpera(allE,allT,newW)[0]==0:
                    ans.append(max(findbestpera(allE,allT,newW)[3]))   
        for i in range(len(dataset_50_train)):
            if dataset_50_train[1][i][4]==ans[i]:
                c+=1
        return (c/len(dataset_50_train))

    def CalaccDepth(i,ent,rt):
        print("DEPTH = ",i)
        print("Accuracy | Train = ",ent," Test = ",rt)
            

    Supportcalaccdepth()   

    print('END Q1_AB\n')


if __name__ == "__main__":
    i=3
    main(i)
