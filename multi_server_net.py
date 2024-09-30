import socket
import threading
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

denny=run_network()

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

def handle_client(client_socket, addr):
    try:
        while True:
            request = client_socket.recv(1024).decode("utf-8")
            if request.lower() == "close":
                client_socket.send("closed".encode("utf-8"))
                break
            if (len(request)!=49):
                client_socket.send("incorrect format".encode("utf-8"))
                continue
            print(f"Received: {request}")
            result = run_test(request,denny)
            response = f"accepted ({addr[0]}:{addr[1]})\n{result}"
            ###
            ###response = f"accepted ({addr[0]}:{addr[1]})" 
            
            client_socket.send(response.encode("utf-8"))
    
    except Exception as e:
        print(f"Error when handling client: {e}")
    finally:
        client_socket.close()
        print(f"Connection to client ({addr[0]}:{addr[1]}) closed")

def run_server():
    server_ip = "127.0.0.1"
    port = 8000
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((server_ip, port))
        server.listen()
        print(f"Listening on {server_ip}:{port}")
        
        while True:
            client_socket, addr = server.accept()
            print(f"Accepted connection from {addr[0]}:{addr[1]}")
            thread = threading.Thread(target=handle_client, args=(client_socket, addr,))
            thread.start()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        server.close()

run_server()