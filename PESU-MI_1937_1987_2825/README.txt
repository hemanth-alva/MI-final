IMPLEMENTATION:

DATA PREPROCESSING:

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

Age has 14 unique values.On plotting a box plot to identify skewness, the Age feature shows slight left skewness. 
So the missing values for Age is filled with the median value.
Delivery Phase shows skewness too. The missing values are filled with the median.
Education has only one unique value 5 and hence the missing values are filled with 5(which happens to be the mode)
Residence shows only 3 unique values and only 2 NaN's and therefore it if filled with 0.
The BP feature is symmetric and therefore the missing values can be directly filled with the mean value.
The HB and the Weight feature shows right skewness and is therefore filled with the median.


BUILDING THE NEURAL NETWORK:

 1) Weight and Bias Initialisation : Weights and Biases are initialised from a random normal distribution.
 2) Input features - There are 9 input features used 
    Target Labels - The 'Result' Column is the target classification label
 3) Forward Propagation - 1) Calculation of the linear combination: WX + b for the first layer
			  2) if intermediate layers:
				Introducing non linearity with the use of the ReLU activation function as it still preserves the desirable properties of linearity
			     if last layer:
				Use the sigmoid activation function that compresses the output to a value between 0 and 1( probability useful for classification) 
			  3) Propagating the results to the next Layer. Back to step 1.
 4) Loss/ Error function - Binary CrossEntropy (Binary Classification)
 5) BackPropagation - BackPropagate and update/nudge the weights in the direction that minimises the loss function through the use of gradient descent.  
 6) Continue with step 3 to 5 for any number of epochs - full batch gradient descent
 7) Plot the loss as a function of epochs
 
HYPERPARAMETERS:

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
Binary Cross Entropy(also known as negative log likelihood function)
BC(y,yhat)=−(ylog(yhat) + (1-y)log(1-yhat)) where y and yhat are probability distributions
y is the actual output classification
yhat is the predicted output classification


KEY FEATURE:
The NN implemented can be tuned to work for any number of layers, any number of neurons, learning rate, and number of epochs.
Further, to ensure that the train and test sets are representative of the inherent distribution(since the dataset is unbalanced), we have used stratified sampling.

ADDITIONAL IMPLEMENTATIONS:
Plot of the Loss Curve:
Plots the loss curve by keeping track of the loss at each epoch.

Comparison Plot:
Plots the accuracy metric for different layer sizes and number of neurons

STEPS TO RUN THE FILE:

cd .\PESU-MI_1937_1987_2825\src
py NNfromscratch.py

