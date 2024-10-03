import streamlit as st
import pandas as pd
import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2 ))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        pass
    
    def backward(self, output_gradient, learning_rate):
        pass
    
class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        derrivitive = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return derrivitive

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class Sigmoid(Activation):
    def __init__(self):
        sig = lambda x: 1/(1+np.exp(-x))
        sig_prime = lambda x: sig(x)*(1-sig(x))
        super().__init__(sig, sig_prime)

#______________________________________________________________________________
def run_network():
    FILE_I = open('TRAIN_I.txt','r')
    FILE_R = open('TRAIN_R.txt','r')
    LINE_I = []
    LINE_R = []

    for line in FILE_I:
        n = list(line)
        if (n[-1] == '\n'):
            n.pop()
        r = 0
        while r < len(n):
            n[r] = float(n[r])
            r += 1

        LINE_I.append(n)
    
    for line in FILE_R:
        n = list(line)
        if (n[-1] == '\n'):
            n.pop()
        r = 0
        while r < len(n):
            n[r] = float(n[r])
            r += 1
        
        LINE_R.append(n)

    INPUT_DIM = 49
    HIDDEN_Q = 3
    HIDDEN_N = 10
    EPOCHS = 1000


    X=np.reshape(LINE_I, (30,49,1))
    Y=np.reshape(LINE_R, (30,3,1))


    network = []
    network.append(Dense(INPUT_DIM, HIDDEN_N))
    network.append(Sigmoid())
    counter = HIDDEN_Q-1

    while counter>0:
        network.append(Dense(HIDDEN_N, HIDDEN_N))
        network.append(Sigmoid())
        counter-=1
    
    network.append(Dense(HIDDEN_N, 3))
    network.append(Sigmoid())

    epochs = EPOCHS
    learning_rate = 0.1

    for e in range(epochs):
        error = 0
    
        for x, y in zip(X, Y):
            output = x
            for layer in network:
                output = layer.forward(output)
            
        
            error += mse (y, output)
        
            grad = mse_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

    
        error /= len(x)
        print('%d/%d, error=%f' % (e+1,epochs,error))
    return network
#______________________________________________________________________________

def run_test(data,denny):
    network = denny
    FILE_T = data
    LINE_T = []
    
    n = list(data)
    
    if (n[-1] == '\n'):
        n.pop()
        
    r = 0
    while r < len(n):
        n[r] = float(n[r])
        r += 1
        
    LINE_T.append(n)

    Z = np.reshape(LINE_T,(1,49,1))
    for z in Z:
        output1=z
        for layer in network:
            output1 = layer.forward(output1)

    pred_fig = np.argmax(output1)
    fig_names = ['circle','square','triangle']
    res = f"Figure is {fig_names[pred_fig]}"
    return(res)

net = run_network()

st.header("Guess the figure")
data = st.text_input("Enter the text representation of a figure")
if st.button("Start"):
    data = str(data)
    if (len(data) == 49):
        try:
            st.write("Wait up! Im thinking...")
            result = run_test(data, net)
            st.subheader(result)
        except Exception as e:
            st.write(f"There's been an error! {e}")
    else:
        st.write("Wrong format buddy!")
        

