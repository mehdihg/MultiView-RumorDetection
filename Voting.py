# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 11:37:58 2021

@author: Rayangostar
"""

"""Barebones general Tree & Node"""
import os
import numpy as np
import pandas as pd
seed=7
import keras
np.random.seed(seed)
import tensorflow as tf
#from tensorflow import keras
from keras import layers
from keras.layers import Dense, Input, GRU,Conv1D,Conv2D,MaxPooling1D,LSTM, Embedding, Dropout, Activation 
from keras.layers import Bidirectional,concatenate
from keras.models import Model  
from keras.models import Sequential
from keras import initializers, regularizers, constraints, optimizers 
from keras.layers import Flatten, BatchNormalization 
from keras.optimizers import Adam,SGD
from keras.wrappers.scikit_learn import KerasClassifier 
from keras.utils import np_utils 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import KFold 
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences 
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot

#import matplotlib.pyplot as plt
num1=0
num2=0
num3=0
num4=0
numallnode=0
maxctimee=180*56
df2= pd.read_excel(r"/content/shufflefilenew2.xlsx") 
countminute=round(float(maxctimee))/20
files2=os.listdir('/content/nontrue15162/')
files=[]
print(files2[0][0:-4],str(df2.values[0][1]))
for i in range(0,len(df2)):
    for j in range(0,len(files2)):
        if(str(df2.values[i][1]) in str(files2[j])):
            print(files2[j])
            files.append(files2[j][0:-4])
            break
print(len(files))
subtree=np.zeros((len(files),int(countminute),4))
#label=np.zeros((len(files),4))
tweet=[]
label=[]

subtreefeature=np.zeros((len(files),int(countminute),12))
class node(object):
    def __init__(self):
        self.name=None
        self.node=[]
        self.otherInfo = None
        self.prev=None
        self.time=None
    def nex(self,child):
        "Gets a node by number"
        return self.node[child]
    def prevv(self):
        return self.prev
    
    def goto(self,data,data2,data3):
        breaker=False
        match=False
        "Gets the node by name"
        for child in range(0,len(self.node)):
            if(self.node[child].name==data):
                match=True
                cnew=self.node[child].add()
                cnew.name=data2
                cnew.time=data3
                return cnew.name,match
                break
        if(match==False):
            for child in range(0,len(self.node)):
                for child2 in range(0,len(self.node[child].node)):
                     if(self.node[child].node[child2]):   
                        if(self.node[child].node[child2].name==data):
                            match=True
                            cnew=self.node[child].node[child2].add()
                            cnew.name=data2
                            cnew.time=data3
                            return cnew.name,match
                            break
                if(breaker):
                    break
   
        if(match==False):
            for child in range(0,len(self.node)):
                for child2 in range(0,len(self.node[child].node)):
                    for child3 in range(0,len(self.node[child].node[child2].node)):
                        if(self.node[child].node[child2].node[child3]):   
                            if(self.node[child].node[child2].node[child3].name==data):
                                match=True
                                cnew=self.node[child].node[child2].node[child3].add()
                                cnew.name=data2
                                cnew.time=data3
                                return cnew.name,match
                                break
                    if(breaker):
                        break
                if(breaker):
                    break         
                                
        if(match==False):
            for child in range(0,len(self.node)):
                for child2 in range(0,len(self.node[child].node)):
                    for child3 in range(0,len(self.node[child].node[child2].node)):
                        for child4 in range(0,len(self.node[child].node[child2].node[child3].node)):
                            if(self.node[child].node[child2].node[child3].node[child4]):   
                                if(self.node[child].node[child2].node[child3].node[child4].name==data):
                                    match=True
                                    cnew=self.node[child].node[child2].node[child3].node[child4].add()
                                    cnew.name=data2
                                    cnew.time=data3
                                    return cnew.name,match
                                    breaker=True
                                    break
                        if(breaker):
                            break
                    if(breaker):
                        break
                if(breaker):
                     break   
                             
       
    def add(self):
        node1=node()
        self.node.append(node1)
        node1.prev=self
        return node1
    


#for child in range(0,int(countminute)):

for file in range(0,len(files)):
    #filee=open('G:\\MODARES\\seminar\\data\\rumdetect2017\\rumor_detection_acl2017\\twitter15\\tree200\\'+files[file],'r')
    filee=open('/content/nontrue15162/'+str(files[file])+'.txt','r')
    line=filee.readline()
    line=filee.readline()
    pindex1 = line.find('[',0,3)
    pindex2 = line.find(',',0,18)
    cindex1 = line.find('>',20,60)
    cindex2 = line.find(',',40,70)
    ctimeindex1=line.find(',',70,95)
    ctimeindex2=line.find(']',65,100)
    pname=line[pindex1+2:pindex2-1]
    cname=line[ cindex1+3: cindex2-1]
    ctime=line[ ctimeindex1+3: ctimeindex2-1]
    
    t=node() 
     #name it root
    t.name=str(int(files[file]))
    tree=t.add()
    numallnode=numallnode+1  
    tree.name=pname
    tree.time=0.0
    t.goto(pname,cname,ctime)
    numallnode=numallnode+1 
    maxctime=0
    for line in filee:
        pindex1 = line.find('[',0,3)
        pindex2 = line.find(',',0,18)
        cindex1 = line.find('>',20,60)
        cindex2 = line.find(',',40,70)
        if( line.find(',',80,105)>80):
            ctimeindex1=line.find(',',80,110)
        else:
            ctimeindex1=line.find(',',65,85)
        ctimeindex2=line.find(']',65,130)
        pname=line[pindex1+2:pindex2-1]
        cname=line[ cindex1+3: cindex2-1]
        ctime=line[ ctimeindex1+3: ctimeindex2-1]
        t.goto(pname,cname,ctime)
        numallnode=numallnode+1
        if(float(maxctime)<float(ctime)):
           maxctime=ctime 
    #labelfile=open('G:\\MODARES\\seminar\\data\\rumdetect2017\\rumor_detection_acl2017\\twitter15\\label.txt','r')
    labelfile=open('/content/labelf.txt','r')
    source_tweet=open('/content/source_tweetsf.txt','r')
    for lf in labelfile:
        if(str(int(files[file])) in lf):
            if('non-rumor'in lf):
                #endchar=lf.find(":")
                #label.append(lf[0:endchar])
                label.append('non-rumor')
                break
            else:
                label.append('rumor')
                break   
    for st in source_tweet:
        if(str(int(files[file])) in st):
             startchar=st.find('\t',1,20)
             tweet.append(st[startchar+1:])
             break       
    maxctime=60
    #countminute=round(float(maxctimee))/20
    for i in range(0,int(countminute)):
        cc=0
        ccc=0
        cccall=0
        ccall=0
        for child in range(0,len(t.node[0].node)):
            cccall2=0
            if(20*i <float(t.node[0].node[child].time) <= 20*(i+1)):
                subtree[file][i][0]=subtree[file][i][0]+1
            c=0
            ccall=0
            for child2 in range(0,len(t.node[0].node[child].node)):
                cccall=0
                if(20*i < float(t.node[0].node[child].node[child2].time) <= 20*(i+1)):   
                    c=1
                    subtree[file][i][1]=subtree[file][i][1]+1
            
                cc=0  
                for child3 in range(0,len(t.node[0].node[child].node[child2].node)):
                    if(20*i < float(t.node[0].node[child].node[child2].node[child3].time) <= 20*(i+1)):   
                        cc=1
                        ccall=1
                        subtree[file][i][2]=subtree[file][i][2]+1
                        
                    ccc=0    
                    for child4 in range(0,len(t.node[0].node[child].node[child2].node[child3].node)):
                        if(20*i < float(t.node[0].node[child].node[child2].node[child3].node[child4].time) <= 20*(i+1)):   
                            ccc=1
                            cccall=1
                            cccall2=0
                            subtree[file][i][3]=subtree[file][i][3]+1     
                    if(ccc==1 and float(t.node[0].node[child].node[child2].node[child3].time)<20*i ):
                        subtree[file][i][2]=subtree[file][i][2]+1    
                if((cc==1 and float(t.node[0].node[child].node[child2].time)<20*i) or (cccall==1 and float(t.node[0].node[child].node[child2].time)<20*i)):
                    subtree[file][i][1]=subtree[file][i][1]+1
                    
            if((c==1 and float(t.node[0].node[child].time)<20*i) or (ccall==1 and float(t.node[0].node[child].time)<20*i) or (cccall2==1 and float(t.node[0].node[child].time)<20*i)):
                subtree[file][i][0]=subtree[file][i][0]+1          
        subtreefeature[file][i][0]=subtree[file][i][0]
        subtreefeature[file][i][1]=subtree[file][i][1]
        subtreefeature[file][i][2]=subtree[file][i][2]
        subtreefeature[file][i][3]=subtree[file][i][3]
        subtreefeature[file][i][4]=subtree[file][i][0]/(subtree[file][i][0]+subtree[file][i][1]+subtree[file][i][2]+subtree[file][i][3]+1)
        subtreefeature[file][i][5]=subtree[file][i][1]/(subtree[file][i][0]+subtree[file][i][1]+subtree[file][i][2]+subtree[file][i][3]+1)
        subtreefeature[file][i][6]=subtree[file][i][2]/(subtree[file][i][0]+subtree[file][i][1]+subtree[file][i][2]+subtree[file][i][3]+1)
        subtreefeature[file][i][7]=subtree[file][i][3]/(subtree[file][i][0]+subtree[file][i][1]+subtree[file][i][2]+subtree[file][i][3]+1)
        if(i>0):
            subtreefeature[file][i][8]=(subtree[file][i][0]+subtree[file][i-1][0])/(subtree[file][i][0]+subtree[file][i][1]+subtree[file][i][2]+subtree[file][i][3]+subtree[file][i-1][0]+subtree[file][i-1][1]+subtree[file][i-1][2]+subtree[file][i-1][3]+1)
            subtreefeature[file][i][9]=(subtree[file][i][1]+subtree[file][i-1][1])/(subtree[file][i][0]+subtree[file][i][1]+subtree[file][i][2]+subtree[file][i][3]+subtree[file][i-1][0]+subtree[file][i-1][1]+subtree[file][i-1][2]+subtree[file][i-1][3]+1)
            subtreefeature[file][i][10]=(subtree[file][i][2]+subtree[file][i-1][2])/(subtree[file][i][0]+subtree[file][i][1]+subtree[file][i][2]+subtree[file][i][3]+subtree[file][i-1][0]+subtree[file][i-1][1]+subtree[file][i-1][2]+subtree[file][i-1][3]+1)
            subtreefeature[file][i][11]=(subtree[file][i][3]+subtree[file][i-1][3])/(subtree[file][i][0]+subtree[file][i][1]+subtree[file][i][2]+subtree[file][i][3]+subtree[file][i-1][0]+subtree[file][i-1][1]+subtree[file][i-1][2]+subtree[file][i-1][3]+1)
        else:
            subtreefeature[file][i][8]=subtreefeature[file][i][4]
            subtreefeature[file][i][9]=subtreefeature[file][i][5]
            subtreefeature[file][i][10]=subtreefeature[file][i][6]
            subtreefeature[file][i][11]=subtreefeature[file][i][7]
    print(file,subtreefeature[file][0][10],subtreefeature[file][1][6],files[file],startchar,countminute,label[file])        
tweets=np.array(tweet)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)
word_index = tokenizer.word_index
le = LabelEncoder() 
le.fit(label) 
labels = le.transform(label)
labels = keras.utils.to_categorical(np.asarray(labels))  
MAX_SEQUENCE_LENGTH=max([len(s.split()) for s in tweets])
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
 
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from sklearn import preprocessing
from keras.layers import Dense
import random
# load the dataset
#df = pd.read_excel(r"F:\\centralityf25.xlsx")
df = pd.read_excel(r"/content/centstructtime.xlsx")
# split into input (X) and output (y) variables
X = df.values[:,3:]
X = np.asarray(X).astype('float32')
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1,8))
minmax = min_max_scaler.fit_transform(X)
X=minmax
X=np.reshape(X,(X.shape[0], 1,X.shape[1])) 
yy= df.values[:,2]
print(yy[0:5])
le = LabelEncoder() 
le.fit(yy) 
y = le.transform(yy)
y = keras.utils.to_categorical(np.asarray(y)) 
print(y[0:5])
X_test=X[917:]
y_test=y[917:]

model1 = load_model('/content/best_models7913.hdf5')
model1.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
pre1=model1.predict(subtreefeature[917:])
score = model1.evaluate(subtreefeature[917:], y_test, verbose=0)
print("%s: %.2f%%" % (model1.metrics_names[1], score[1]*100))

model2 = load_model('best_modeln8000.hdf5')
model2.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
score = model2.evaluate(X[917:], labels[917:], verbose=0)
print("%s: %.2f%%" % (model2.metrics_names[1],score[1]*100))
pre2=model2.predict(X[917:])

model3 = load_model('best_modelc8782.hdf5')
model3.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
score = model3.evaluate(data[917:], labels[917:], verbose=0)
print("%s: %.2f%%" % (model3.metrics_names[1],score[1]*100))
pre3=model3.predict(data[917:])
#print(labels[0:10],y_test[0:10])
#print(pre1[0:10],pre2[0:10])
preout=[]
lpre1=[]
lpre2=[]
lpre3=[]
lpreout=[]
testarg=[]
for i in range(0,len(labels[917:])):
  preout.append([0.5*pre1[i][0]+0.5*pre2[i][0]+0.5*pre3[i][0],0.5*pre1[i][1]+0.5*pre2[i][1]+0.5*pre3[i][1]])
  lpre1.append(np.argmax(pre1[i], axis = 0))
  lpreout.append(np.argmax(preout[i], axis = 0))
  lpre2.append(np.argmax(pre2[i], axis = 0))
  lpre3.append(np.argmax(pre3[i], axis = 0))
  testarg.append(np.argmax(y_test[i], axis = 0))
print(preout[0:10],"\n",lpre1[0:10],"\n",lpre2[0:10])
print(testarg[0:10],"\n",lpreout[0:10])
a=0
for i in range(0,len(y_test)):
  if(testarg[i] != lpreout[i]):
    a=a+1
print(a)
print('ACC=',(float(len(labels[917:]))-a)/float(len(labels[917:]))*100)
TR=0
TN=0
FR=0
FN=0
for i in range(0,len(y_test)):
    if(lpreout[i]==testarg[i] and lpreout[i]==1):
        TR=TR+1
    elif(lpreout[i]==testarg[i] and lpreout[i]==0):
        TN=TN+1
    elif(lpreout[i]!=testarg[i] and lpreout[i]==0):
        FN=FN+1
    elif(lpreout[i]!=testarg[i] and lpreout[i]==1):
        FR=FR+1
        
preR=TR/(TR+FR)
preN=TN/(TN+FN)
reR=TR/(TR+FN)
reN=TN/(TN+FR)
FR1=2*((preR*reR)/(preR+reR))
FN1=2*((preN*reN)/(preN+reN))
print(TR,TN,FR,FN,'preR',preR,'preN',preN,'reR',reR,'reN',reN,'FR1',FR1,'FN1',FN1)
