'''
NN Architechture: 

Total number of layers used: 4 
Number of Hidden Layers: 3
Output layer: 1
The number of perceptrons at the input: 9 (9 input features on which training takes place)
The number of perceptrons at the hidden layer 1: 7
The number of perceptrons at the hidden layer 2: 5
The number of perceptrons at the hidden layer 3: 8
The number of perceptrons at the output layer: 1 (Binary output (0 or 1))

The learning rate: 0.001
The number of epochs: 800
Weight Initialisation: Random Normal Distribution
Optimisation: Batch Gradient Descent
Code Capable of adapting to any number of layers

Dimensions of the Weight Matrices: 
W1: 9 x 7 (input and hidden layer 1)
W2: 7 x 5 (hidden layer 1 and hidden layer 2)
W3: 5 x 8 (hidden layer 2 and hidden layer 3)
W4: 8 x 1 (hidden layer 3 and the output layer(layer 4))

Dimensions of the Bias Matrices:
B1: 7 x 1 (bias for each of the 7 perceptrons in the hidden layer 1)
B2: 5 x 1 (bias for each of the 5 perceptrons in the hidden layer 2)
B3: 8 x 1 (bias for each of the 8 perceptrons in the hidden layer 3)
B4: 1 x 1 (bias for the perceptron in the output layer)

Activation Function:
Hidden Layer 1: ReLU
Hidden Layer 2: ReLU
Hidden Layer 3: ReLU
Output Layer: Sigmoid (To get the output to a probabililty(sum is 1) that classifies the input to either of the 2 classes)

Loss Function:
Binary Cross Entropy( also known as negative log likelihood function)
BC(y,yhat)=−(ylog(yhat) + (1-y)log(1-yhat)) where y and yhat are probability distributions
y is the actual output classification
yhat is the predicted output classification

Plot of the Loss Curve:
Plot the loss curve by keeping track of the loss at each epoch

Data Preprocessing:
The dataset has missing values with the distribution as given below:
Community          0
Age                7
Weight            11
Delivery phase     4
HB                19
IFA                0
BP                15
Education          3
Residence          2
Result             0
dtype: int64

Age has 14 unique values.On plotting a box plot to identify skewness, the Age feature shows slight left skewness. So the missing values for Age is filled
with the median value.
Delivery Phase shows skewness too. The missing values are filled with the median.
Education has only one unique value 5 and hence the missing values are filled with 5(which happens to be the mode)
Residence shows only 3 unique values and only 2 NaN's and therefore it if filled with 0.
The BP feature is symmetric and therefore the missing values can be directly filled with the mean value.
The HB and the Weight feature shows right skewness and is therefore filled with the median.


Sigmoid Activation Function and Binary Cross entropy Loss Functions:
Take care of overflow and underflow conditions and add a small lambda for yhat=0 or yhat=1 to handle log(0) errors

Forward Propagation:
For each layer i:
    Calculate y = input*weights + bais for layer
    if layer = last layer:
        yhat = sigmoid(y)
        break
    else:
        input  = relu(y)
Calculate the Binary CrossEntropy loss 

Backpropagation:
BC(y,yhat)= −(ylog(yhat) + (1-y)log(1-yhat))
derivative of BC wrt to W(FinalHiddenLayer) = derivative of BC wrt to yhat * derivative of yhat wrt net * derivative of net wrt W(FinalHiddenLayer) 
derivative of BC wrt to yhat = - (y/yhat - (1-y)/(1-yhat))
derivative of yhat wrt net = derivative of ( 1/1+exp(-net)) wrt net = yhat*(1-yhat)
derivative of net wrt W(FinalHiddenLayer) = A = (input to the last layer)

'''


import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NN:

    ''' X and Y are dataframes '''
    def __init__(self, layers=[9,7,5,8,1], lr=0.001, epochs=1000):
            self.layers = layers
            self.lr = lr
            self.epochs = epochs
            self.lossBC = []
            self.input = None
            self.label = None
            self.parameters = dict()
               
    def InitialiseWeights(self,nooflayers):
            '''Weight Initialisation from a random normal distribution
            parameters['Wx'] for x in [1,2,..,layers] defines the weight matrix for layer x
            parameters['bx'] for x in [1,2,..,layers] defines the bias matrix for layer x
            '''

            np.random.seed(1) # Seed the random number generator
            for layer in range(0,nooflayers-1):
                self.parameters['W'+str(layer+1)] = np.random.randn(self.layers[layer], self.layers[layer+1])
                self.parameters['b'+str(layer+1)] = np.random.randn(self.layers[layer+1],)

    def Relu(self,Y):
        ''' ReLU for intermediate layers: Positive thresholding'''
        return np.maximum(0, Y)
        
        
    def Sigmoid(self,Y):
        ''' Sigmoid to convert it to a range beteen 0 and 1'''

        '''code to prevent overflow and underflow'''
        res =np.zeros(Y.shape,dtype='float')
        for i in range(0,len(Y)):
            if -Y[i] > np.log(np.finfo(float).max):
                res[i]= 0.0 
            elif Y[i] > np.log(np.finfo(float).max):
                res[i] = 0.0
            else:
                res[i]=1.0 / (1.0 + np.exp(-Y[i])) 
        return res
        
    def BinaryCrossEntropyLoss(self,y,yhat):
        '''Binary Cross Entropy Loss Function
        y is the actual output classification
        yhat is the predicted output classification
        '''
        number = len(y)
        #index = np.where(n_array == 0)[0]
        #print(yhat.shape , y.shape)
        '''code to take care of log(0) by adding a small lambda value '''
        for i in range(len(yhat)): 
            if yhat[i]==0:
                yhat[i]= 1e-7
            elif yhat[i]==1:
                yhat[i] -= 1e-7
        #print(yhat)
        '''Binary CrossEntropy Function'''
        loss = -1/number * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat)))) 
        return loss

    def ReluDerivative(self,x):
        ''' Derivative of the ReLU function'''
        x[x<=0] = 0
        x[x>0] = 1
        return x
    
    def ForwardPropagation(self,nooflayers):
        '''Forward Propagation 
           nooflayers - number of layers in the NN
           input - features used for training
           L stores input*W + b (linear combination)
           A stores the result of activation applied to L that is given as the input to the next layer
        '''
        L1 = self.input.dot(self.parameters['W1']) + self.parameters['b1']
        A1 = self.Relu(L1)
        self.parameters['L1'] = L1
        self.parameters['A1'] = A1
        for i in range(2,nooflayers-1):
            L1 = A1.dot(self.parameters['W'+str(i)]) + self.parameters['b'+str(i)]
            A1 = self.Relu(L1)
            self.parameters['L'+str(i)] = L1
            self.parameters['A'+str(i)] = A1
        L2 = A1.dot(self.parameters['W'+str(nooflayers-1)]) + self.parameters['b'+str(nooflayers-1)]
        #print(Z2.shape , self.y.shape)
        yhat = self.Sigmoid(L2)
        #print(yhat.shape)
        loss = self.BinaryCrossEntropyLoss(self.label,yhat)
        self.parameters['L'+str(nooflayers-1)] = L2

        return yhat,loss

    
    def BackPropagation(self,yhat,nooflayers):
        '''Backward Propagation that also performs updating of the weights simultaneously
            yhat - the predicted output classification 
            diffyhat - derviative of loss/error wrt yhat
            diffsigmoid - derviative of yhat wrt to non-activated previous linear combination (sigmoid derviative)
            diffz - diffyhat * diffsigmoid
        '''
        #print(yhat)
        diffyhat = -(np.divide(self.label,yhat) - np.divide((1 - self.label),(1-yhat)))
        #dl_wrt_yhat[np.isnan(dl_wrt_yhat)]=0.0
        #print(dl_wrt_yhat)
        '''Sigmoid derivative for the last layer'''
        diffsigmoid = yhat * (1-yhat) 
        diffz = diffyhat * diffsigmoid

        for i in range(nooflayers-1,1,-1):
            #print(self.params['A'+str(i-1)].T,dl_wrt_z2)
            diffA1 = diffz.dot(self.parameters['W'+str(i)].T)
            diffw2 = self.parameters['A'+str(i-1)].T.dot(diffz)
            diffb2 = np.sum(diffz, axis=0)
            
            '''updation of the weights and biases for intermediate layers'''
            self.parameters['W'+str(i)] = self.parameters['W'+str(i)] - self.lr * diffw2
            self.parameters['b'+str(i)] = self.parameters['b'+str(i)] - self.lr * diffb2

            #print(self.params['W'+str(i)])
            #print(self.params['A'+str(i-1)], (1 - self.params['A'+str(i-1)]))
            diffrelu = self.ReluDerivative(self.parameters['A'+str(i-1)]) 
            #print(dl_wrt_yhat , dl_wrt_sig)
            diffz = diffA1 * diffrelu


        diffz1 = diffA1 * self.ReluDerivative(self.parameters['L1'])
        diffw1 = self.input.T.dot(diffz1)
        diffb1 = np.sum(diffz1, axis=0)

        '''updation of the weights and bias for layer 1'''
        self.parameters['W1'] = self.parameters['W1'] - self.lr * diffw1
        self.parameters['b1'] = self.parameters['b1'] - self.lr * diffb1

    def fit(self,X,y):
        '''Model Training
        X - inpit features for training
        y - label for classification output
        '''
        #np.seterr(all='raise')
        self.input = X
        self.label = y
        '''initialize weights and bias'''
        self.InitialiseWeights(len(self.layers)) 


        for i in range(self.epochs):
            yhat, loss = self.ForwardPropagation(len(self.layers))
            self.BackPropagation(yhat,len(self.layers))
            #print(loss)
            self.lossBC.append(loss)
            
                                                            
    def predict(self,X):
        '''Predicting on Test Data'''
        for i in range(1,len(self.layers)):
            L1 = X.dot(self.parameters['W'+str(i)]) + self.parameters['b'+str(i)]
            A1 = self.Relu(L1)
            X=A1
        predicted = self.Sigmoid(L1)
        yhat = np.round(predicted)  
        return yhat

    def accuracy(self, y, yhat):
        '''Accuracy as a Performance Metric'''
        accuracy = (sum(y == yhat) / len(y) * 100)
        return accuracy


    def BCLossPlot(self):
        ''' Plot of Loss Function '''
        plt.title("Loss Plot")
        plt.plot(self.lossBC)
        plt.xlabel("No of Epochs")
        plt.ylabel("Loss")
        
        plt.show()

    def CM(self,y_test,y_test_obs):
        '''
        Prints confusion matrix 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0

        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0

        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)

        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")


def ComparisonPlot():
    '''ADDITIONAL COMPONENT
    Plot that compares different layer sizes and number of neurons for accuracy'''
    li = [[9,5,5,3,1],[9,6,8,1],[9,10,10,1],[9,7,7,7,1],[9,7,5,8,1]]
    KS = len(li)
    j = 0
    mean_acc =np.zeros((KS))
    for i in li:
        #print(i)
        nn =  NN(layers= i, lr=0.001, epochs=1000)
        nn.fit(Xtrain, ytrain)
        test_pred = nn.predict(Xtest)
        mean_acc[j] = nn.accuracy(ytest,test_pred)
        #print(nn.acc(ytest,test_pred))
        j += 1
    
        #print(nn.acc(ytest,test_pred))

    print("The best accuracy is", mean_acc.max(), "with layer values =",li[mean_acc.argmax()])
    #mean_acc
    mi = []
    for i in li :
        #print(i)
        mi.append(str(i))
    plt.figure(figsize=(9, 5))
    plt.title("Comparison Plot")
    plt.ylabel("Accuracy")
    plt.xlabel("Layer Values")
    plt.bar(mi,mean_acc)
    plt.show()


data = pd.read_csv("../data/CleanedLBW_Dataset.csv")
#print(data.shape)
X = data.drop(columns=['Result'])
y = data['Result'].values.reshape(X.shape[0], 1)

'''Train Test Split: Stratified Distribution Splitting To evaluate the model '''
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state = 42, stratify = y)
'''Scaling/ Normalising the Data'''
sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

#print(Xtrain.shape , ytrain.shape, Xtest.shape , ytest.shape)
'''building the NN model'''
nn1 = NN(layers=[9,7,5,8,1], lr=0.001, epochs=800)
'''training the model to fit the data'''
nn1.fit(Xtrain, ytrain) 
nn1.BCLossPlot()

trainPrediction = nn1.predict(Xtrain)
testPrediction = nn1.predict(Xtest)

print("Train accuracy : {}".format(nn1.accuracy(ytrain, trainPrediction)))
print("Test accuracy : {}".format(nn1.accuracy(ytest, testPrediction)))
nn1.CM(ytest, testPrediction)

'''Uncomment the following statement to observe the comparision plot'''
#ComparisonPlot()
