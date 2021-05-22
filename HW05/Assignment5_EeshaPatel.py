import numpy as np 
   
X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0,1],[1,0],[1,0],[0,1]), dtype=float)

def sigmoid(t):
    return 1/(1+np.exp(-t))

def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(self.input.shape[1],6)
        self.weights2 = np.random.rand(6,2)
        self.y = y
        self.output = np. zeros(y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2
        
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2*(self.y -self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2*(self.y -self.output)*
                                                 sigmoid_derivative(self.output), self.weights2.T)*
                                                 sigmoid_derivative(self.layer1))
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
        

NN = NeuralNetwork(X,y)
for i in range(1000):
    if i % 200 ==0: 
        print ("For iteration #" + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))
        print ("\n")
  
    NN.train(X, y)

X = np.array(([0,0,0],[1,1,1]), dtype=float)
h1 = sigmoid(np.dot(X, NN.weights1))
y_pred = sigmoid(np.dot(h1, NN.weights2))
print ("The prediction is: \n" + str(y_pred))