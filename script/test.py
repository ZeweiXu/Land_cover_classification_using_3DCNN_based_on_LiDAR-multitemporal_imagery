from thundersvm import *
from sklearn import preprocessing
from sklearn.datasets import *
import time
import os
os.chdir('/home/cc/thundersvm/python/')
import sys
# GPU ID
gpuID = int(sys.argv[1])
# load training data
x =  np.load('training_data.npy')#[:,:66]
y = np.load('training_label.npy')
# training data reshuffling and normalization
from sklearn.utils import shuffle
x, y  = shuffle(x, y)
std_scale = preprocessing.StandardScaler().fit(x)
x = std_scale.transform(x)
# Voting function
def prerefine(prer,testlen,num):
    import collections
    tela=[]
    testlen -= num
    temp = prer[:num]
    pre = prer[num:]
    for i in range(testlen):
        counter=collections.Counter(pre[i*9:(i*9+9)])
        va=counter.values()
        tela.extend([counter.keys()[va.index(int(max(va)))]])
    tela = np.array(tela)
    temp = np.concatenate((temp,tela),axis = 0)
    return temp
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
# Model training
clf = SVC(verbose=False,gpu_id=gpuID,gamma=4.40212623e-03,C=4.5256)
clf.fit(x,y)
# Model prediction
x2 = np.load('testing_data.npy')#[:,:66]
x2 = std_scale.transform(x2)
y_p2=clf.predict(x2)
# accuracy evaluation
y2 = np.load('testing_label9.npy')
yt = np.load('testing_label.npy')
num = np.where(y2 == 1)[0][-1] + 1
npre = prerefine(y_p2,len(yt),num)
ee2=confusion_matrix(yt,npre)
ee2=np.transpose(ee2)
print('Confusion matrix is: ')
print(ee2)
print('Overall accuracy is:')
print(str(np.trace(ee2)/float(len(yt))))
