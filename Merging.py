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
from keras.callbacks import ModelCheckpoint,EarlyStopping
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
#import matplotlib.pyplot as plt
num1=0
num2=0
num3=0
num4=0
numallnode=0
maxctimee=180*56
df2= pd.read_excel(r"/content/MultiView-RumorDetection/shufflefilenew2.xlsx") 
countminute=round(float(maxctimee))/20
files2=os.listdir('/content/MultiView-RumorDetection/nontrue15162/')
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
    filee=open('/content/MultiView-RumorDetection/nontrue15162/'+str(files[file])+'.txt','r')
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
    labelfile=open('/content/MultiView-RumorDetection/labelf.txt','r')
    source_tweet=open('/content/MultiView-RumorDetection/source_tweetsf.txt','r')
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
opt=Adam(lr=0.002)
# structuremodel = Sequential()
# structuremodel.add(Bidirectional(GRU(6, kernel_initializer='normal', input_shape=(int(countminute), 12), activation='tanh', dropout=0.2)))
# structuremodel.add(Dense(2, activation='softmax'))
# structuremodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# structuremodel.fit(
#     subtreefeature[0:803], labels[0:803],
#     validation_data=(subtreefeature[803:917], labels[803:917]),
#     batch_size=10, epochs=150,
#     callbacks=[
#         EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
#         ModelCheckpoint("MultiView-RumorDetection/best_models7913.hdf5", monitor='val_accuracy', save_best_only=True)
#     ]
# )
early_stopping_net = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
structuremodel = Sequential()
structuremodel.add(GRU(6, kernel_initializer='normal', input_shape=(int(countminute), 12), activation='tanh', dropout=0.2))
structuremodel.add(Dense(2, activation='softmax'))
structuremodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

structuremodel.fit(
    subtreefeature[0:803], labels[0:803],
    validation_data=(subtreefeature[803:917], labels[803:917]),
    batch_size=10, epochs=2,
    callbacks=[ModelCheckpoint("MultiView-RumorDetection/best_models7913.hdf5", monitor='val_accuracy', save_best_only=True),
    early_stopping_net
    ]
)





import numpy as np

from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model, Sequential

from transformers import TFBertModel, BertTokenizer, create_optimizer

import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense, Dropout, concatenate, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# --- 1. آماده‌سازی ورودی ---
MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH    # یا مقدار قبلی شما

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded = bert_tokenizer(
    tweets.tolist(),
    max_length=MAX_SEQUENCE_LENGTH,
    padding='max_length',
    truncation=True,
    return_tensors='tf'
)
input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']

# --- 2. ساخت مدل ---
ids_in  = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')
mask_in = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')

# بارگذاری و فعال‌سازی BERT
bert_encoder = TFBertModel.from_pretrained('bert-base-uncased')
bert_encoder.trainable = True

bert_out = bert_encoder(ids_in, attention_mask=mask_in)
sequence_output = bert_out.last_hidden_state  # (batch_size, seq_len, hidden_size)

# CNN با دو کرنل
conv3 = Conv1D(128, 3, activation='relu', padding='same')(sequence_output)
conv5 = Conv1D(128, 5, activation='relu', padding='same')(sequence_output)
cnn_concat = concatenate([conv3, conv5], axis=-1)  # (batch, seq_len, 256)

# Multi-Head Attention (num_heads=4, key_dim=64)
mha = MultiHeadAttention(num_heads=4, key_dim=64)(cnn_concat, cnn_concat)
mha_norm = LayerNormalization()(mha)

# Pooling + Dense layers
x = GlobalMaxPooling1D()(mha_norm)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)

# خروجی
outputs = Dense(2, activation='softmax', name='context_output')(x)

# مدل نهایی
contexmodel = Model(inputs=[ids_in, mask_in], outputs=outputs, name='bert_cnn_mha')

# --- 3. کامپایل ---
contexmodel.compile(
    optimizer=Adam(learning_rate=3e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# --- 4. آموزش ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ModelCheckpoint('MultiView-RumorDetection/best_modeln8000.hdf5',
                    monitor='val_accuracy', save_best_only=True)
]

contexmodel.fit(
    [input_ids[:803], attention_mask[:803]],
    labels[:803],
    validation_data=([input_ids[803:917], attention_mask[803:917]], labels[803:917]),
    batch_size=8,
    epochs=50,
    callbacks=callbacks
)

# --- 5. ارزیابی مدل ---
loss, accuracy = contexmodel.evaluate(
    [input_ids[803:917], attention_mask[803:917]], labels[803:917], verbose=0
)
print(f"\n✅ مدل روی داده‌ی اعتبارسنجی:\n  Accuracy: {accuracy * 100:.2f}%\n  Loss: {loss:.4f}")

y_true = labels[803:917].argmax(axis=1)
y_pred_probs = contexmodel.predict([input_ids[803:917], attention_mask[803:917]])
y_pred = y_pred_probs.argmax(axis=1)

# گزارش‌ها
print("📌 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# ماتریس سردرگمی
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - BERT-CNN-MHA")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# AUC (فقط برای binary)
if y_pred_probs.shape[1] == 2:
    roc_auc_val = roc_auc_score(y_true, y_pred_probs[:, 1])
    print(f"🔵 ROC AUC: {roc_auc_val:.4f}")



# فقط برای binary classification
fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
roc_auc = auc(fpr, tpr)

precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - BERT-CNN-MHA")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.title("Precision-Recall Curve - BERT-CNN-MHA")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
plt.show()






import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import random

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical




import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, Attention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# --- Load & preprocess data ---
df = pd.read_excel("/content/MultiView-RumorDetection/centstructtime.xlsx")

idd = df.values[:, 1]
X = df.values[:, 3:].astype('float32')

# Scaling (standardization instead of MinMax)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Labels
yy = df.values[:, 2]
le = LabelEncoder()
y = le.fit_transform(yy)
y = to_categorical(y)

# Reshape inputs for Conv1D
X = np.expand_dims(X, axis=2)

# Split
X_train, X_val, X_test = X[0:803], X[803:917], X[917:]
y_train, y_val, y_test = y[0:803], y[803:917], y[917:]

# --- Model with Conv1D + BiLSTM + Attention ---
input_layer = Input(shape=(X.shape[1], 1))

x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Attention()([x, x])  # Self-attention

x = GlobalAveragePooling1D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(2, activation='softmax')(x)

Netmodel = Model(inputs=input_layer, outputs=output_layer)

# Compile
Netmodel.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
checkpoint = ModelCheckpoint("MultiView-RumorDetection/best_model_structtime_attn.hdf5", monitor='val_accuracy', save_best_only=True)

# Train
Netmodel.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=8,
    epochs=150,
    callbacks=[early_stopping, checkpoint]
)

# Evaluate
test_loss, test_accuracy = Netmodel.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")


y_true_net = y_test.argmax(axis=1)
y_pred_net_probs = Netmodel.predict(X_test)
y_pred_net = y_pred_net_probs.argmax(axis=1)

print("📌 Classification Report - Structural-Time:")
print(classification_report(y_true_net, y_pred_net, digits=4))

cm_net = confusion_matrix(y_true_net, y_pred_net)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_net, annot=True, fmt='d', cmap='Oranges')
plt.title("Confusion Matrix - Structural-Time")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

if y_pred_net_probs.shape[1] == 2:
    roc_auc_score_net = roc_auc_score(y_true_net, y_pred_net_probs[:, 1])
    print(f"🔵 ROC AUC: {roc_auc_score_net:.4f}")



fpr_net, tpr_net, _ = roc_curve(y_true_net, y_pred_net_probs[:, 1])
roc_auc_net = auc(fpr_net, tpr_net)

precision_net, recall_net, _ = precision_recall_curve(y_true_net, y_pred_net_probs[:, 1])

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr_net, tpr_net, label=f"ROC Curve (AUC = {roc_auc_net:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Structural-Time")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall_net, precision_net, label="Precision-Recall Curve")
plt.title("Precision-Recall Curve - Structural-Time")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()

plt.tight_layout()
plt.show()












import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, Lambda, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- 0. آماده‌سازی مدل‌های پیش‌آموزش‌دیده ----------------
# (در صورت نیاز لایه‌های آنها را فریز کنید)
for layer in contexmodel.layers:
    layer.trainable = False
for layer in Netmodel.layers:
    layer.trainable = False

# ---------------- 1. یافتن و ساخت مدل‌های استخراج ویژگی ----------------
# 1.1. پیدا کردن لایه Dense(64) در contexmodel
intermediate_bert_layer = None
for layer in reversed(contexmodel.layers):
    if isinstance(layer, Dense) and layer.output_shape[-1] == 64:
        intermediate_bert_layer = layer.name
        break
assert intermediate_bert_layer, "لایه Dense با خروجی 64 در contexmodel پیدا نشد!"

bert_feature_model = Model(
    inputs=contexmodel.input,
    outputs=contexmodel.get_layer(intermediate_bert_layer).output
)

# 1.2. پیدا کردن لایه Dense(64) در Netmodel
intermediate_net_layer = None
for layer in reversed(Netmodel.layers):
    if isinstance(layer, Dense) and layer.output_shape[-1] == 64:
        intermediate_net_layer = layer.name
        break
assert intermediate_net_layer, "لایه Dense با خروجی 64 در Netmodel پیدا نشد!"

net_feature_model = Model(
    inputs=Netmodel.input,
    outputs=Netmodel.get_layer(intermediate_net_layer).output
)

# ---------------- 2. تعریف ورودی‌های مدل نهایی ----------------
# ورودی‌های BERT
input_ids_in, attention_mask_in = contexmodel.input
# ورودی Net
net_input = Netmodel.input

# ---------------- 3. استخراج ویژگی‌ها ----------------
bert_feat = bert_feature_model([input_ids_in, attention_mask_in])  # (None, 64)
net_feat  = net_feature_model(net_input)                          # (None, 64)

# ---------------- 4. Weighted Adaptive Fusion ----------------
# 4.1. گیت ترکیب
fusion_concat = Concatenate(name='fusion_concat')([bert_feat, net_feat])  # (None,128)
gates = Dense(2, activation='softmax', name='fusion_gate')(fusion_concat) 
# gates[:,0] برای BERT، gates[:,1] برای Net

# 4.2. جداسازی و اعمال وزن‌ها
w_bert = Lambda(lambda x: tf.expand_dims(x[:, 0], -1), name='w_bert')(gates)  # (None,1)
w_net  = Lambda(lambda x: tf.expand_dims(x[:, 1], -1), name='w_net')(gates)   # (None,1)

fused = Add(name='fused_features')([
    tf.multiply(w_bert, bert_feat),
    tf.multiply(w_net,  net_feat)
])  # (None,64)

# ---------------- 5. سر مدل (classification head) ----------------
x = Dense(64, activation='relu', name='head_dense')(fused)
x = Dropout(0.3, name='head_dropout')(x)
output = Dense(2, activation='softmax', name='head_output')(x)

# ---------------- 6. ساخت و کامپایل مدل نهایی ----------------
final_model = Model(
    inputs=[input_ids_in, attention_mask_in, net_input],
    outputs=output,
    name='WeightedAdaptiveFusionModel'
)
final_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
final_model.summary()

# ---------------- 7. آماده‌سازی داده‌ها ----------------
# فرض: input_ids, attention_mask، X و labels (one-hot) پیش‌تر بارگذاری شده‌اند.
Xb = [input_ids, attention_mask]
Xn = X
y  = labels

# تقسیم داده‌ها
split1, split2 = 803, 917
Xb_train = [arr[:split1] for arr in Xb]
Xn_train = Xn[:split1]
y_train  = y[:split1]

Xb_val   = [arr[split1:split2] for arr in Xb]
Xn_val   = Xn[split1:split2]
y_val    = y[split1:split2]

Xb_test  = [arr[split2:] for arr in Xb]
Xn_test  = Xn[split2:]
y_test   = y[split2:]

# ---------------- 8. آموزش مدل ----------------
early_stop = EarlyStopping(
    monitor='val_loss', patience=50, restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    "best_weighted_fusion_model.h5",
    monitor="val_accuracy", save_best_only=True
)

history = final_model.fit(
    x = Xb_train + [Xn_train],
    y = y_train,
    validation_data=(Xb_val + [Xn_val], y_val),
    batch_size=8,
    epochs=150,
    callbacks=[early_stop, checkpoint]
)

# ---------------- 9. ارزیابی نهایی ----------------
loss, acc = final_model.evaluate(Xb_test + [Xn_test], y_test, verbose=0)
print(f"\n✅ Final Weighted Adaptive Fusion Accuracy: {acc:.4f}")

y_true       = np.argmax(y_test, axis=1)
y_pred_probs = final_model.predict(Xb_test + [Xn_test])
y_pred       = np.argmax(y_pred_probs, axis=1)

print("\n📌 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))

# ماتریس درهم‌ریختگی
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ROC و PR Curve
auc_score = roc_auc_score(y_true, y_pred_probs[:,1])
print(f"🔵 ROC AUC: {auc_score:.4f}")

fpr, tpr, _     = roc_curve(y_true, y_pred_probs[:,1])
precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:,1])
roc_auc         = auc(fpr, tpr)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1,2,2)
plt.plot(recall, precision, label="Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.show()


















# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # ---------------- 1. استخراج ویژگی‌های میانی از BERT و Net ----------------
# # پیدا کردن Dense(64) از contexmodel
# intermediate_bert_layer = None
# for layer in reversed(contexmodel.layers):
#     if isinstance(layer, Dense) and layer.output_shape[-1] == 64:
#         intermediate_bert_layer = layer.name
#         break
# assert intermediate_bert_layer is not None, "❌ لایه Dense با خروجی 64 در contexmodel پیدا نشد!"

# bert_feature_model = Model(inputs=contexmodel.input, outputs=contexmodel.get_layer(intermediate_bert_layer).output)

# # پیدا کردن Dense(64) از Netmodel
# intermediate_net_layer = None
# for layer in reversed(Netmodel.layers):
#     if isinstance(layer, Dense) and layer.output_shape[-1] == 64:
#         intermediate_net_layer = layer.name
#         break
# assert intermediate_net_layer is not None, "❌ لایه Dense با خروجی 64 در Netmodel پیدا نشد!"

# net_feature_model = Model(inputs=Netmodel.input, outputs=Netmodel.get_layer(intermediate_net_layer).output)

# # پیش‌بینی ویژگی‌ها
# X_bert = bert_feature_model.predict([input_ids, attention_mask])
# X_net = net_feature_model.predict(X)

# # ---------------- 2. ترکیب ویژگی‌ها ----------------
# X_combined = np.concatenate([X_bert, X_net], axis=1)

# # تقسیم داده‌ها
# X_train = X_combined[0:803]
# X_val   = X_combined[803:917]
# X_test  = X_combined[917:]

# y_train_combined = labels[0:803]
# y_val_combined   = labels[803:917]
# y_test_combined  = labels[917:]

# # ---------------- 3. ساخت مدل نهایی ترکیبی ----------------
# final_input = Input(shape=(X_train.shape[1],))
# x = Dense(64, activation='relu')(final_input)
# x = Dropout(0.3)(x)
# final_output = Dense(2, activation='softmax')(x)

# final_model = Model(inputs=final_input, outputs=final_output)
# final_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# # ---------------- 4. آموزش مدل نهایی با ذخیره بهترین نسخه ----------------
# early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
# checkpoint = ModelCheckpoint("MultiView-RumorDetection/best_combined_model.hdf5", monitor="val_accuracy", save_best_only=True)

# final_model.fit(
#     X_train, y_train_combined,
#     validation_data=(X_val, y_val_combined),
#     batch_size=8,
#     epochs=150,
#     callbacks=[early_stop, checkpoint]
# )

# # ---------------- 5. ارزیابی مدل نهایی ----------------
# loss, acc = final_model.evaluate(X_test, y_test_combined)
# print(f"\n✅ Final Combined Model Accuracy: {acc:.4f}")

# y_true = y_test_combined.argmax(axis=1)
# y_pred_probs = final_model.predict(X_test)
# y_pred = y_pred_probs.argmax(axis=1)

# # 📋 گزارش آماری
# print("📌 Classification Report - Combined:")
# print(classification_report(y_true, y_pred, digits=4))

# # 📊 Confusion Matrix
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
# plt.title("Confusion Matrix - Final Combined Model")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()

# # 🔷 ROC و PR Curve
# if y_pred_probs.shape[1] == 2:
#     auc_score = roc_auc_score(y_true, y_pred_probs[:, 1])
#     print(f"🔵 ROC AUC: {auc_score:.4f}")

#     fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
#     precision, recall, _ = precision_recall_curve(y_true, y_pred_probs[:, 1])
#     roc_auc = auc(fpr, tpr)

#     plt.figure(figsize=(12, 5))

#     plt.subplot(1, 2, 1)
#     plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve - Combined Model")
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(recall, precision, label="Precision-Recall")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Precision-Recall Curve - Combined Model")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()










# import pandas as pd
# import numpy as np
# from tensorflow import keras
# from keras.models import Sequential
# from sklearn import preprocessing
# from keras.layers import Dense
# import random
# # load the dataset
# #df = pd.read_excel(r"F:\\centralityf25.xlsx")
# df = pd.read_excel(r"/content/MultiView-RumorDetection/centstructtime.xlsx")
# # split into input (X) and output (y) variables
# idd=df.values[:,1]
# X = df.values[:,3:]
# X = np.asarray(X).astype('float32')
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1,8))
# minmax = min_max_scaler.fit_transform(X)
# X=minmax
# yy= df.values[:,2]
# print(yy[0:5])
# le = LabelEncoder() 
# le.fit(yy) 
# y = le.transform(yy)
# y = keras.utils.to_categorical(np.asarray(y)) 
# print(y[0:5])
# b=0
# a=0    
# X_test=X[917:]
# y_test=y[917:]
# # define the keras model
# Netmodel = Sequential()
# X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
# Netmodel.add(LSTM(8, input_shape=(1, 8), kernel_initializer='uniform', activation='relu'))
# Netmodel.add(Dense(4, kernel_initializer='uniform', activation='relu'))
# Netmodel.add(Dense(2, activation='softmax'))
# Netmodel.compile(optimizer=Adam(lr=0.002), loss='binary_crossentropy', metrics=['accuracy'])

# early_stopping_net = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Netmodel.fit(
#     X[0:803], y[0:803],
#     validation_data=(X[803:917], y[803:917]),
#     batch_size=10, epochs=3,
#     callbacks=[
#         ModelCheckpoint("MultiView-RumorDetection/best_modelsn84347.hdf5", monitor='val_accuracy', save_best_only=True),
#         early_stopping_net
#     ]
# )
np.save("subtreefeature.npy", subtreefeature)
np.save("data.npy",           data)
np.save("X.npy",              X)
np.save("y_test.npy",         y_test)

#opt=SGD(lr=0.05)







#validation_data=([subtreefeature[170:],data[170:]], labels[170:])


# from keras.models import Model, load_model
# from keras.layers import Dense, concatenate
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# import numpy as np
# from transformers import TFBertModel

# # --- 1. merge outputs ---
# merged = concatenate([
#     structuremodel.output,
#     Netmodel.output,
#     contexmodel.output
# ])

# final_output = Dense(2, activation='softmax')(merged)

# # --- 2. build final model with *flat* list of inputs ---
# all_inputs = [structuremodel.input, Netmodel.input] + contexmodel.input
# final_model = Model(inputs=all_inputs, outputs=final_output)

# # --- 3. compile ---
# final_model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # --- 4. callbacks ---
# early_stop_concat = EarlyStopping(
#     monitor='val_loss',
#     patience=30,
#     restore_best_weights=True
# )
# checkpoint_concat = ModelCheckpoint(
#     "MultiView-RumorDetection/best_modelsn.hdf5",
#     monitor='val_accuracy',
#     save_best_only=True
# )

# # --- 5. train (fit) ---
# final_model.fit(
#     # ورودی‌ها: [X_struct, X_net, input_ids, attention_mask]
#     [
#         subtreefeature[0:803],     # X_struct_train
#         X[0:803],                  # X_net_train
#         input_ids[0:803],          # BERT input_ids train
#         attention_mask[0:803]      # BERT attention_mask train
#     ],
#     labels[0:803],
#     validation_data=(
#       [
#         subtreefeature[803:917],
#         X[803:917],
#         input_ids[803:917],
#         attention_mask[803:917]
#       ],
#       labels[803:917]
#     ),
#     batch_size=10,
#     epochs=150,
#     callbacks=[early_stop_concat, checkpoint_concat]
# )

# # --- 6. save final model ---
# final_model.save("MultiView-RumorDetection/best_modelsn.hdf5")

# # --- 7. load & predict on test set ---
# concat_best = load_model(
#     "MultiView-RumorDetection/best_modelsn.hdf5",
#     custom_objects={'TFBertModel': TFBertModel}
# )

# # prepare test inputs
# X_struct_test = subtreefeature[917:]
# X_net_test    = X[917:]
# ids_test      = input_ids[917:]
# mask_test     = attention_mask[917:]

# y_pred_concat = np.argmax(
#     concat_best.predict([X_struct_test, X_net_test, ids_test, mask_test]),
#     axis=1
# )

# # گزارش عملکرد
# report("Final Concatenated Model", y_true, y_pred_concat)

# # ذخیره پیش‌بینی‌ها
# np.save("predictions_concat.npy", y_pred_concat)




# model = Model(inputs=[structuremodel.input, Netmodel.input,contexmodel.input[0],  # input_ids
#         contexmodel.input[1]], outputs=final_output)
# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.002), metrics=['accuracy'])


# early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# model.fit(
#     [
#         subtreefeature[:803],
#         X[:803],
#         input_ids[:803],
#         attention_mask[:803]
#     ],
#     labels[:803],
#     validation_data=(
#         [subtreefeature[803:917], X[803:917], input_ids[803:917], attention_mask[803:917]],
#         labels[803:917]
#     ),
#     batch_size=8,
#     epochs=50,  # تعداد ایپاک‌ها را به 20 تغییر دهید
#     callbacks=[
#         tf.keras.callbacks.ModelCheckpoint(
#             "MultiView-RumorDetection/best_modelsn.hdf5", monitor='val_accuracy', save_best_only=True
#         ),
#         early_stopping  # اضافه کردن EarlyStopping
#     ]
# )



# model.fit(
#     [subtreefeature[0:803], X[0:803], data[0:803]],
#     labels[0:803],
#     validation_data=([subtreefeature[803:917], X[803:917], data[803:917]], labels[803:917]),
#     batch_size=10, epochs=150,
#     callbacks=[ModelCheckpoint("MultiView-RumorDetection/best_modelsn.hdf5", monitor='val_accuracy', save_best_only=True)]
# )
#model.fit([subtreefeature[0:803]], labels[0:803],callbacks=[checkpoint], batch_size=10, epochs=100,validation_data=([subtreefeature[803:917]],labels[803:917]))#,shuffle=True

print(countminute)
# from transformers import TFBertModel
# from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization

# from keras.models import load_model
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# # --- 1. بارگذاری مدل‌ها ---
# structure_best = load_model('MultiView-RumorDetection/best_models7913.hdf5')


# contex_best = tf.keras.models.load_model(
#     'MultiView-RumorDetection/best_modeln8000.hdf5',
#     custom_objects={
#         'TFBertModel': TFBertModel,
#         'MultiHeadAttention': MultiHeadAttention,
#         'LayerNormalization': LayerNormalization
#     }
# )
# net_best       = load_model('MultiView-RumorDetection/best_modelsn84347.hdf5')

# # --- 2. آماده‌سازی داده‌های آزمون ---
# subtreefeature = np.load("subtreefeature.npy")
# data            = np.load("data.npy")
# X               = np.load("X.npy")
# y_test          = np.load("y_test.npy")

# X_struct = subtreefeature[917:]  # داده‌های ویژگی مربوط به ساختار
# X_text   = data[917:]           # داده‌های ویژگی مربوط به متن
# X_net    = X[917:]              # داده‌های ویژگی مربوط به شبکه
# y_true   = np.argmax(y_test, axis=1)  # تبدیل one-hot به لیبل عددی

# # --- 3. پیش‌بینی هر مدل ---
# pred_s = np.argmax(structure_best.predict(X_struct), axis=1)
# pred_c = np.argmax(contex_best.predict([input_ids[917:], attention_mask[917:]]), axis=1)
# pred_n = np.argmax(net_best.predict(X_net), axis=1)

# # --- 4. تابع کمکی برای چاپ گزارش ---
# def report(name, y_true, y_pred):
#     print(f"===== {name} =====")
#     print("Accuracy :", accuracy_score(y_true, y_pred))
#     print("Precision:", precision_score(y_true, y_pred, average='binary'))
#     print("Recall   :", recall_score(y_true, y_pred, average='binary'))
#     print("F1-Score :", f1_score(y_true, y_pred, average='binary'))
#     print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
#     print("\n")

# # --- 5. چاپ گزارش برای هر مدل ---
# report("Structure-Model (GRU)", y_true, pred_s)
# report("Context-Model (BERT)", y_true, pred_c)
# report("Centrality-Model (LSTM)", y_true, pred_n)

# # --- 6. ترکیب پیش‌بینی‌ها با روش Voting ---
# # Majority voting
# vote_preds = []
# for i in range(len(pred_s)):
#     votes = [pred_s[i], pred_c[i], pred_n[i]]
#     vote_preds.append(max(set(votes), key=votes.count))

# # --- 7. گزارش ترکیب Voting ---
# report("Multi-View Ensemble Voting", y_true, vote_preds)

# # --- 8. ذخیره‌سازی پیش‌بینی‌ها برای ارزیابی‌های بیشتر ---
# np.save("predictions_s.npy", pred_s)
# np.save("predictions_c.npy", pred_c)
# np.save("predictions_n.npy", pred_n)
# np.save("vote_predictions.npy", vote_preds)


















# lpre=[]
# testarg=[]
# TR=0
# TN=0
# FR=0
# FN=0
# from keras.models import load_model
# merge_loaded = load_model('MultiView-RumorDetection/best_modelsn.hdf5', custom_objects={'TFBertModel': TFBertModel})
# loss, acc = merge_loaded.evaluate([subtreefeature[917:], X[917:], input_ids[917:], attention_mask[917:]], y_test, verbose=0)
# print(f"Merge Model Test loss: {loss:.4f}, accuracy: {acc:.4f}")
# pre_preds = np.argmax(merge_loaded.predict([subtreefeature[917:], X[917:], input_ids[917:], attention_mask[917:]]), axis=1)
# true_labels = np.argmax(y_test, axis=1)
# TR = np.sum((pre_preds == true_labels) & (pre_preds == 1))
# TN = np.sum((pre_preds == true_labels) & (pre_preds == 0))
# FR = np.sum((pre_preds != true_labels) & (pre_preds == 1))
# FN = np.sum((pre_preds != true_labels) & (pre_preds == 0))
# print(f"TP: {TR}, TN: {TN}, FP: {FR}, FN: {FN}")

# # --- 5. Voting Ensemble of Individual Views ---
# # Load best checkpoints of three separate views
# structure_best = load_model('MultiView-RumorDetection/best_models7913.hdf5')
# contex_best    = load_model(
#     'MultiView-RumorDetection/best_modeln8000.hdf5',
#     custom_objects={'TFBertModel': TFBertModel}
# )
# net_best       = load_model('MultiView-RumorDetection/best_modelsn84347.hdf5')

# # Predictions on test set
# pred_s = np.argmax(structure_best.predict(subtreefeature[917:]), axis=1)
# pred_c = np.argmax(contex_best.predict([input_ids[917:], attention_mask[917:]]), axis=1)
# pred_n = np.argmax(net_best      .predict(X[917:]),              axis=1)

# # Majority voting
# vote_preds = []
# for i in range(len(pred_s)):
#     votes = [pred_s[i], pred_c[i], pred_n[i]]
#     vote_preds.append(max(set(votes), key=votes.count))

# # Compute voting metrics
# TR = TN = FR = FN = 0
# for i, true in enumerate(true_labels):
#     pred = vote_preds[i]
#     if pred == true:
#         if pred == 1: TR += 1
#         else:         TN += 1
#     else:
#         if pred == 1: FR += 1
#         else:         FN += 1

# accuracy_vote = (TR + TN) / len(true_labels)
# preR = TR / (TR + FR) if (TR + FR) > 0 else 0
# preN = TN / (TN + FN) if (TN + FN) > 0 else 0
# reR  = TR / (TR + FN) if (TR + FN) > 0 else 0
# reN  = TN / (TN + FR) if (TN + FR) > 0 else 0
# F1R  = 2 * preR * reR / (preR + reR) if (preR + reR) > 0 else 0
# F1N  = 2 * preN * reN / (preN + reN) if (preN + reN) > 0 else 0

# print(f"Voting Accuracy: {accuracy_vote*100:.2f}%")
# print(f"TP: {TR}, TN: {TN}, FP: {FR}, FN: {FN}")
# print(f"Precision(rumor): {preR:.3f}, Recall(rumor): {reR:.3f}, F1(rumor): {F1R:.3f}")
# print(f"Precision(non-rumor): {preN:.3f}, Recall(non-rumor): {reN:.3f}, F1(non-rumor): {F1N:.3f}")

# # Save feature arrays for reproducibility
# np.save("subtreefeature.npy", subtreefeature)
# np.save("data.npy",           data)
# np.save("X.npy",              X)
# np.save("y_test.npy",         y_test)






# from keras.models import load_model

# import numpy as np

# # --- 1. بارگذاری مدل‌ها ---
# structure_best = load_model('MultiView-RumorDetection/best_models7913.hdf5')
# contex_best    = load_model(
#     'MultiView-RumorDetection/best_modeln8000.hdf5',
#     custom_objects={'TFBertModel': TFBertModel}
# )
# net_best       = load_model('MultiView-RumorDetection/best_modelsn84347.hdf5')

# # --- 2. آماده‌سازی داده‌های آزمون ---
# # subtreefeature[917:], data[917:], X[917:] و y_test از کد شما

# subtreefeature = np.load("subtreefeature.npy")
# data            = np.load("data.npy")
# X               = np.load("X.npy")
# y_test          = np.load("y_test.npy")

# # حالا ببرید مثل قبل:
# X_struct = subtreefeature[917:]
# X_text   = data[917:]
# X_net    = X[917:]
# y_true   = np.argmax(y_test, axis=1)



# X_struct = subtreefeature[917:]
# X_text   = data[917:]
# X_net    = X[917:]        # بعد از reshape: (num_samples, 1, feature_dim)
# y_true   = np.argmax(y_test, axis=1)  # تبدیل one-hot به لیبل عددی

# # --- 3. پیش‌بینی هر مدل ---
# pred_s = np.argmax(structure_best.predict(X_struct), axis=1)
# pred_c = np.argmax(contex_best.predict([input_ids[917:], attention_mask[917:]]), axis=1)
# pred_n = np.argmax(net_best      .predict(X_net  ), axis=1)

# # --- 4. تابع کمکی برای چاپ گزارش ---
# def report(name, y_true, y_pred):
#     print(f"===== {name} =====")
#     print("Accuracy :", accuracy_score(y_true, y_pred))
#     print("Precision:", precision_score(y_true, y_pred, average='binary'))
#     print("Recall   :", recall_score   (y_true, y_pred, average='binary'))
#     print("F1-Score :", f1_score       (y_true, y_pred, average='binary'))
#     print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))
#     print("\n")

# # --- 5. چاپ گزارش برای هر مدل ---
# report("Structure-Model (GRU)", y_true, pred_s)
# report("Context-Model (BERT)",   y_true, pred_c)
# report("Centrality-Model (LSTM)", y_true, pred_n)

# # (اختیاری) اگر مدل ترکیبی را هم دارید:
# ensemble_best = load_model('MultiView-RumorDetection/best_modelsn.hdf5', custom_objects={'TFBertModel': TFBertModel})
# pred_e = np.argmax(
#     ensemble_best.predict([X_struct, X_net, input_ids[917:], attention_mask[917:]]),
#     axis=1
# )
# report("Multi-View Ensemble Concat", y_true, pred_e)