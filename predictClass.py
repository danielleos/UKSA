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

#transform scaled data csv to matrix
df = scaleTest.as_matrix()
#if the (standardised) class value is less than 0, colour datapoint red, otherwise blue
#TNA against Cryptonine-3
#plt.plot(scaleTest['TNA'],scaleTest['Cryptonine-3'])
df = scaleTest.as_matrix()
for i in range(0,len(scaleTest)):
    if (df[i][13] < 0):
        #red - class 0
        plt.plot(df[i][0], df[i][3], 'r.')
    else:
        #blue - class 1
        plt.plot(df[i][0], df[i][3], 'b.')

#c3 only
df = scaleTest.as_matrix()
for i in range(0,len(scaleTest)):
    if (df[i][13] < 0):
        #red - 0
        plt.plot(i, df[i][3], 'r.')
    else:
        #blue - 1
        plt.plot(i, df[i][3], 'b.')
	
#TNA only
for i in range(0,len(scaleTest)):
    if (df[i][13] < 0):
        #red - 0
        plt.plot(i, df[i][0], 'r.')
    else:
        #blue - 1
        plt.plot(i, df[i][0], 'b.')

#entropy function
# We will be using class colour-coded plot results from above 
# in the decision tree building process with appropriate thresholds
def entropy(dataFrame, threshold, attribute, tvalue, tlabel):
    df = dataFrame.copy()
    #find column numbers (for referencing matrix in loops - can't iterate over a dataFrame)
    aColumn = df.columns.get_loc(attribute)
    tColumn = df.columns.get_loc(tlabel)
    dm = df.as_matrix()
    
    # 'counters'
    np1 = 0
    nn1 = 0
    np2 = 0
    nn2 = 0
    
    #iterate over each row
    for i in range(0,len(df)):
        #check if a datapoint is above threshold for specified attribute (in our case, above 0 for TNA)
        if dm[i][aColumn] > threshold:
            #check if that same point is of class 0
            if dm[i][tColumn] <= tvalue:
                np1 += 1
            #or of class 1
            elif dm[i][tColumn] > tvalue:
                nn1 += 1
        #otherwise if it's below the threshold
        elif dm[i][aColumn] <= threshold:
            if dm[i][tColumn] <= tvalue:
                #add to a separate counter (since we've split our data into two halves)
                np2 += 1
            elif dm[i][tColumn] > tvalue:
                nn2 += 1
                
    #sum up our instances in each half
    n1 = np1 + nn1
    n2 = np2 + nn2
    
    #first half
    if np1 == 0 or nn1 == 0:
        entropy1 = 0
    else:
        #definition of entropy
        entropy1 = -((np1/n1)*math.log((np1/n1),2) + (nn1/n1)*math.log((nn1/n1),2))
    
    #second half
    if np2 == 0 or nn2 == 0:
        entropy2 = 0
    else:
        entropy2 = -((np2/n2)*math.log((np2/n2),2) + (nn2/n2)*math.log((nn2/n2),2))

    return entropy1, entropy2

#entropy(scaleTest, 0, 'TNA', 0, 'Class')
#output: roughly (0.746, 0.882)

#computes entropy for binary/discrete classes (and for only two classes - this would need adapting for multiclass examples)
def singleEntropy(dataFrame, aClass, bClass, tlabel):
    df = dataFrame.copy()
    na = (df[tlabel] <= aClass).sum()
    nb = (df[tlabel] >= bClass).sum()
    n = na + nb
    
    if na == 0 or nb == 0:
        entropy = 0
    else:
        entropy = -((na/n)*math.log((na/n),2) + (nb/n)*math.log((nb/n),2))

    return entropy

#information gain function
#still needs adapting...
def infoGain(dataFrame, threshold, attribute, tvalue, tlabel):
    df = dataFrame.copy()
    dm = df.as_matrix()
    sEntropy = entropy(dataFrame, threshold, attribute, tvalue, tlabel)
    aColumn = df.columns.get_loc(attribute)
    tColumn = df.columns.get_loc(tlabel)
    sSize = len(dataFrame)
    sumEnt = 0

    for i in dataFrame[attribute].unique():
        siSize = (dataFrame[attribute] == i).sum()
        sumEnt += (siSize/sSize)*entropy(dataFrame,threshold, attribute, tvalue, tlabel)
    
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
