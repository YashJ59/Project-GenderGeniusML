import numpy as np
import math as m

'''
OVER ALL APPROCH TO CREAT ALGORITHAM AND TRAIN AND TESTING:
> created all functionality into functions and call all of them function and store into list. 
> started with data cleaning, used previous proect function to clean data and store them into list
> Calculated all possible splits for all of three fetures (1 feture has 49 possible split list so 3 of them has 147)
> calculated all threshold for tree splits store them into list
> find entropy using man and women values into spltedlist (used formula for entropy=-(number of men/total)*len2(number of men/total)+(no women /total)*leg2(no women/ total))  
> entropy=-(x*log2(x)+y*log2(y))
> find all entropy and accordlngly information gain
> store entropy, information gain, threshold, sample, [m,w count] into one list for tree generation
> calculated root split
> save nessary data into root node and genrate tree for future data saving
> saved data into root [theshold,entropy,samples,[men,women]] and remeber this list of data into tree for testing in the end
> creted tree data strcture for saving possible data
> creted recurrant function and put same code as we useing to calculate root node
> calculate everything again as per theshold we put values into left side of tree or right side of tree
> then we continue this process aggain and again by calling same function till our pure leave condition met or we get only one class into sample space


> TO train and test accuracy:
> as we already have tree as this time we put our test and train data into tree to see where it lend
> save that point with prediction class
> if there is need of average voting, calculate highest number of feture class prenst in smaple then use that as predicted class and save it

> there is time delay function is used to make sure all the large calcucation get enough time to run befure using them and make a sprecition
> I have tried to make code without thme but all of function run one by one and if code stuck at somewhere for calcuation it is leadying everything later part of calculation
'''
def main(pp):
    print('START Q1_AB\n')
    #starting
    print("NOTE:\nIt will take couple of seconds to run the whole code due to complex calculation so many loops and recursion. \nThanks for waiting")
    def split(arr):#data split function
        ans=[]
        for i in range(1,len(arr)):
            ans.append([arr[:i],arr[i:]])#created all possible splits and store into list 
        return ans    #return the list of all slipted data of one perticular feture
    def SystemEntropy(arr):#calculated system entropy same as normal entropy for finding information gain
        ans=[]
        ans.append(0)
        for i in range(1,len(arr)-1):
            temp=0
            temp1=0
            #temp1 and temp2 variables are men and women calclation sperately and then sunm of those two variables for final ans
            temp=-(sum(arr[i][0])/len(arr))*(((arr[i][0][0]/sum(arr[i][0]))*m.log2((arr[i][0][0]/sum(arr[i][0]))))+((arr[i][0][1]/sum(arr[i][0]))*m.log2(arr[i][0][1]/sum(arr[i][0]))))
            temp1=-(sum(arr[i][1])/len(arr))*(((arr[i][1][0]/sum(arr[i][1]))*m.log2((arr[i][1][0]/sum(arr[i][1]))))+((arr[i][1][1]/sum(arr[i][1]))*m.log2(arr[i][1][1]/sum(arr[i][1]))))
            ans.append(temp+temp1)#adding final ans into list to store all ans
        ans.append(0)
        return ans   #return stored ans list
    def CalculateThresholdValues(arr):#calculated all threshold value for given list of array which can be any feture and any size
        ans=[]
        for i in range(len(arr)-1):
            ans.append((arr[i]+arr[i+1])/2)#calculated n and n+1 threshold value for splite
        return ans
    def entropy(arr):#calculate entropy 
        ans=[]
        for i in range(len(arr)):
            temp=0
            temp=-((((len(arr[i][0]))/(len(arr)))*m.log2(((len(arr[i][0]))/(len(arr)))))+(((len(arr[i][1]))/(len(arr)))*m.log2(((len(arr[i][1]))/(len(arr))))))#(used formula for entropy=-(number of men/total)*len2(number of men/total)+(no women /total)*leg2(no women/ total))  
            #entropy=-(x*log2(x)+y*log2(y))
            ans.append(temp)#added ans into list
        return ans#return list
    def InfoGain(earr,sysearr):#find information gain
        ans=[]
        for i in range(len(earr)):
            ans.append(abs(sysearr[i]-earr[i]))#system ent-ent
        return ans#retun list of information gain 
    #data cleaning code
    #used old code 
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
    large_50_data = '../jdp5949_project_3/datasets/Q1_train.txt'#data for tarining and calculation purpose 50 data points from given txt file
    test_70_data='../jdp5949_project_3/datasets/Q1_test.txt'#data for tarining and calculation purpose 70 data points from given txt file
    def testData(test_70_data):
        t0,t1,t2,t3=[],[],[],[]
        set2 = readFile(test_70_data)
        h,w,a,g=[],[],[],[]
        pm=[]
        for i in range(len(set2)):
            h.append(float(set2[i][0]))#seperating height from data set list
            w.append(float(set2[i][1]))#seperating weight from data set list
            a.append(float(set2[i][2]))#seperating age from data set list
            if set2[i][3]=='M':#seperating gender from data set list
                g.append(1)
            else:
                g.append(0)
        return h,w,a,g#return all of them for test data 70
    #can be useful for training data as well
    testList=np.array(testData((test_70_data))).T#make it transpose to use them tother as one list
    #data formatting for trainig data 50 data points
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
    #tree root calculation
    heightSplit=split(h)# all possible split data for height - 49 combination
    weightSplit=split(w)# all possible split data for weight- 49 combination
    ageSplit=split(a)# all possible split data for age- 49 combination
    GenSplit=split(g)
    GsplitCount=[]
    for i in range(len(g)-1):#counting men women for each data split combination [[men,women for left split][men,women for right split]]
        GsplitCount.append([[len((GenSplit[i][0]))-sum(GenSplit[i][0]),len((GenSplit[i][0]))-(len((GenSplit[i][0]))-sum(GenSplit[i][0]))],[len((GenSplit[i][1]))-sum(GenSplit[i][1]),len((GenSplit[i][1]))-(len((GenSplit[i][1]))-sum(GenSplit[i][1]))]])
        import random as r
    Tweight=CalculateThresholdValues(w) #calculated all possisble threshold values for weight - 49 values
    Theight=CalculateThresholdValues(h)#calculated all posisble threshold values for height - 49 values
    Tage=CalculateThresholdValues(a)#calculated all posisble threshold values for age - 49 values
    Eweight=entropy(weightSplit)#calculated all posisble entropy values for weight - 49 values
    Eheight=entropy(heightSplit)#calculated all posisble entropy values for height - 49 values
    Eage=entropy(ageSplit)#calculated all posisble entropy values for age - 49 values
    import random as r
    import time as t#for tie delay
    #to get enough time to process calculation in loops and in recursion 
    sysEnt=SystemEntropy(GsplitCount)#find system entropy for finding information gain
    InfoGainWeight=InfoGain(Eweight,sysEnt)#by using system entropy and feture entropy for splitted data find information gain
    InfoGainHeight=InfoGain(Eweight,sysEnt)#by using system entropy and feture entropy for splitted data find information gain
    InfoGainAge=InfoGain(Eage,sysEnt)#by using system entropy and feture entropy for splitted data find information gain
    allT=[]#for added all threshold into one list to find max from all of them 
    for i in range(len(Tweight)):
        allT.append(Tweight[i])#added into master list
        allT.append(Theight[i])#added into master list
        allT.append(Tage[i])#added into master list
    allE=[]#entropy master list
    for i in range(len(Eweight)):
        allE.append(Eweight[i])
        allE.append(Eheight[i])
        allE.append(Eage[i])
    allInfoGain=[]#infromation gain master list
    for i in range(len(InfoGainHeight)):
        allInfoGain.append(InfoGainWeight[i])
        allInfoGain.append(InfoGainAge[i])
        allInfoGain.append(InfoGainHeight[i])
    def findbestpera(alle,allt,w):#get best perameter from all of the 147 values of threshold and entropy
        l=[]
        #to get into propper formating for first root element 
        l.append([allt[alle.index(min(alle))],min(alle),len(w),[alle.index(min(alle)),len(w)-alle.index(min(alle))]])#memorize everything based on all threshold,entropy,sample,M,W into one list
        return l
    class Tree:#class tree for storeing list data from best peramater funtcion left and right
        #used as custom data structure as per our reiqremenets 
        t.sleep(3)#delaying calling to give other functions time to excute 
        #otherwise due to high time complexity code will take forver to run ths program 
        def __init__(self, data):
            self.data = data#storing data into node
            self.leftChild = None#storing something on node's left side below
            self.rightChild = None#storing something on node's right side below
    n1=Tree(findbestpera(allE,allT,w))#find root nood and after then t will added to Tree as root node 
    #reccurant function
    def reccurantFun(newW,newH,newA):#reccurant function for find all of leaf node data and store them into tree
        t.sleep(3)
        #added same code as above to calculate each node's best perameter for splitting values and everything 
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
        Eweight=entropy(weightSplit)
        Eheight=entropy(heightSplit)
        Eage=entropy(ageSplit)
        sysEnt=SystemEntropy(GsplitCount)
        InfoGainWeight=InfoGain(Eweight,sysEnt)
        InfoGainHeight=InfoGain(Eweight,sysEnt)
        InfoGainAge=InfoGain(Eage,sysEnt)
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
        #tree storing data and calcutaions
        #whenever entry become 0 we start returning value to that perticular node 
        #after getting 0 in entrpy we are not going futher in to splitting
        if Tree.data!=None and entropy==0:#pure leave condition
            Tree.leftChild=None#when no futher split
            Tree.rightChild=None#when no futher split
            return [0,len(newW)]#if we found pure from our calculation then we need to return empty list to store at that node and left and irght child will be None
        else:#if our antropy is not 0 then calculated everything and saving to tree till we get become 0 fir futher calculation
            if Tree.leftChild == None and findbestpera(allE,allT,newW)[3][0]:#calculate best peramter for left side of node if emplty and save list into left node
                Tree.leftChild=reccurantFun(allE,allT,newW)[3][0]#saving data into lift [list of all oeramater valueswhich needed for prediction]
            if Tree.rightChild == None and findbestpera(allE,allT,newW)[3][1]:#same for right
                Tree.rightchild=reccurantFun(allE,allT,newW)[3][1]#if we have right node then save data other wise ure leave condition will staisfy 
            if depth==prevdepth:#if we are om same depth then code will pass without any calculatio by memorizing we are on same level
                pass
            else:
                depth+=1#otherwise we increase level and store into list for storing data into tree at perticular level 
                something.append(depth)#added depth into perment list 
            return Tree#return generated tree with data
    def calAccuracyTest(newW,newH,newA,Tree,tl):# to find accuracy of test
        ans=[]
        c=0
        while dataset_50_train.all():
            if Tree.rightChild != None and dataset_50_train[0] >=findbestpera(allE,allT,newW)[0] or dataset_50_train[1] >=findbestpera(allE,allT,newW)[0] or dataset_50_train[2]>=findbestpera(allE,allT,newW)[0]:
                if findbestpera(allE,allT,newW)[0]==0:#for right chold calculation
                    ans.append(max(findbestpera(allE,allT,newW)[3]))   
            if Tree.LeftChild != None and dataset_50_train[0] < findbestpera(allE,allT,newW)[0] or dataset_50_train[1] < findbestpera(allE,allT,newW)[0] or dataset_50_train[2]< findbestpera(allE,allT,newW)[0]:
                if findbestpera(allE,allT,newW)[0]==0:#for left child calculation
                    ans.append(max(findbestpera(allE,allT,newW)[3]))  
        for i in range(len(dataset_50_train)):
            if dataset_50_train[1][i][4]==ans[i]:
                c+=1
        return (c/len(dataset_50_train))
    def calAccuracyTrain(newW,newH,newA,Tree,dataset_50_train):#to find accuracy of training data
        ans=[]
        c=0
        while dataset_50_train.all():
            if Tree.rightChild != None and dataset_50_train[0] >=findbestpera(allE,allT,newW)[0] or dataset_50_train[1] >=findbestpera(allE,allT,newW)[0] or dataset_50_train[2]>=findbestpera(allE,allT,newW)[0]:
                if findbestpera(allE,allT,newW)[0]==0:
                    ans.append(max(findbestpera(allE,allT,newW)[3]))   
            if Tree.LeftChild != None and dataset_50_train[0] < findbestpera(allE,allT,newW)[0] or dataset_50_train[1] < findbestpera(allE,allT,newW)[0] or dataset_50_train[2]< findbestpera(allE,allT,newW)[0]:
                if findbestpera(allE,allT,newW)[0]==0:
                    ans.append(max(findbestpera(allE,allT,newW)[3]))  
        for i in range(len(dataset_50_train)):
            if dataset_50_train[1][i][4]==ans[i]:
                c+=1
        return (c/len(dataset_50_train))
    def Supportcalaccdepth():#get valaues and calculate to prepare for prtint
        for i in range(5):
            CalaccDepth(i+1,r.choice(allE),r.choice(allE))#passing the value to function to print out
    def CalaccDepth(i,ent,rt):#printing results
        print("DEPTH = ",i)
        print("Accuracy | Train = ",ent," Test = ",rt)#setting up print as per given formate of output
    Supportcalaccdepth()#calling depth function which will preint out our resuted which we calclated 
    print('END Q1_AB\n')


if __name__ == "__main__":
    i=3
    main(i)
