import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import math
from __future__ import division
import sklearn.linear_model as lm

%matplotlib inline

#read in csv files
classA = pd.read_csv('classA.csv', header=0)
probeA = pd.read_csv('probeA.csv', header=0)
probeB = pd.read_csv('probeB.csv', header=0)

#create single dataset with class labels and training flags
#append class label and random assignment of training flags
trainData = np.append(probeA,classA,axis=1)
trainFlag = np.random.choice([0,1], size=(1000,1), p=[1./3,2./3])
trainData = np.append(trainData,trainFlag,axis=1)

#list of headers (not including training flags)
headers = ['TNA', 'c1', 'c2', 'c3', 'm1', 'm2', 'm3', 'n1', 'n2', 'n3', 'p1', 'p2', 'p3', 'class']

#save as new csv file
np.savetxt("trainData.csv", trainData, delimiter=",")
dataFrame = pd.read_csv('trainData.csv', header=None, \
                        names=['TNA', 'c1', 'c2', 'c3', 'm1', 'm2', 'm3', 'n1', 'n2', 'n3', 'p1', 'p2', 'p3', \
                               'class', 'train'])

#print dataFrame

#create training data set and standardise - preprocessing function
#x is input csv, headers = list of headers, tlabel = target (class in our case)
#isTrain = flag for training dataset
def scaleData(dataFrame, headers):
	df = dataFrame.copy()
	dm = df.as_matrix()
    
    column = 0
    #means and stds for all columns
    for header in headers:
        mean = df[header].mean()
        std = df[header].std()
        for row in range(0,len(dm)):
            dm[row][column] = (dm[row][column]-mean)/std
        column += 1
    
    #suppress scientific form
    np.set_printoptions(suppress=True)
    #save new data as csv
    np.savetxt("scaledData.csv", dm, delimiter=",")
    #read in new csv with scaled data
    scaledData = pd.read_csv('scaledData.csv', header=None, \
                             names=['TNA', 'Cryptonine-1', 'Cryptonine-2', 'Cryptonine-3','Mermaidine-1',\
                                    'Mermaidine-2', 'Mermaidine-3','Neraidine-1', 'Neraidine-2', 'Neraidine-3', \
                                    'Posidine-1', 'Posidine-2', 'Posidine-3', 'Class', 'Train'])
    return scaledData

#scaleTest = scaleData(dataFrame, headers)

### Plots to Visualise Data ###

#class_c1
#plt.plot(scaledProbe['Cryptonine-1'],scaledProbe['Class'], 'bo')
#plt.ylabel('Class')
#class_c2
#plt.plot(scaledProbe['Cryptonine-2'], scaledProbe['Class'], 'ro')
#plt.ylabel('Class')
#class_m1
#plt.plot(scaledProbe['Mermaidine-1'], scaledProbe['Class'], 'co')
#plt.ylabel('Class')
#class_m3
#plt.plot(scaledProbe['Mermaidine-3'], scaledProbe['Class'], 'yo')
#plt.ylabel('Class')
#class_n1
#plt.plot(scaledProbe['Neraidine-1'], scaledProbe['Class'], 'ko')
#plt.ylabel('Class')
#class_n2
#plt.plot(scaledProbe['Neraidine-2'], scaledProbe['Class'], 'bv')
#plt.ylabel('Class')
#class_p1
#plt.plot(scaledProbe['Posidine-1'], scaledProbe['Class'], 'cv')
#plt.ylabel('Class')
#class_TNA
#plt.plot(scaledProbe['TNA'], scaledProbe['Class'], 'k^')
#plt.ylabel('Class')

#entropy function
#From class_x plots, we can calculate entropy and information gain
#We will be using these results in the decision tree building process with appropriate thresholds
def entropy(dataFrame, pos_val, neg_val,tLabel):
    npos = (dataFrame[tLabel] == pos_val).sum()
    nneg = (dataFrame[tLabel] == neg_val).sum()
    n = npos + nneg
    
    if npos == 0 or nneg == 0:
        entropy = 0
    else:
        entropy = -((npos/n)*math.log((npos/n),2) + (nneg/n)*math.log((nneg/n),2))

    return entropy
	
#information gain function
def infoGain(dataFrame, attr, pos_val, neg_val, tLabel):
    sEntropy = entropy(dataFrame,pos_val, neg_val, tLabel)
    sSize = len(dataFrame)
    sumEnt = 0

    for i in dataFrame[attr].unique():
        siSize = (dataFrame[attr] == i).sum()
        sumEnt += (siSize/sSize)*entropy(dataFrame.loc[dataFrame[attr]==i],pos_val,neg_val,tLabel)
    
    infogain = sEntropy - sumEnt
    return infogain

def getRsquared(model,training,testing,tLabel):
    model.fit(training.drop(tLabel,axis=1),training[tLabel])
    r2 = model.score(testing.drop(tLabel,axis=1),testing[tLabel])
    print r2

scaledDataFrame = scaleData(dataFrame, headers)

trainingData = scaledDataFrame[scaledDataFrame['Train']==1].copy()
trainingData = trainingData.drop('Train',axis=1)

testData = scaledDataFrame[scaledDataFrame['Train']==0].copy()
testData = testData.drop('Train',axis=1)

#OLS model - using default parameters
lrModel = lm.LinearRegression()
print "Linear Regression: "
getRsquared(lrModel,trainingData,testData,'Class')

#Ridge model - use CV variant to set best regularisation strength
ridgeModel = lm.RidgeCV(alphas=(0.1,1.0,10.0),cv = 10)
print "Ridge Regression: "
getRsquared(ridgeModel,trainingData,testData,'Class')

#Lasso model
lassoModel = lm.LassoCV()
print "Lasso Regression: "
getRsquared(lassoModel,trainingData,testData,'Class')
