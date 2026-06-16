from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler  # CHANGED: removed LabelEncoder import
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
import numpy as np 
import time
import matplotlib.pyplot as plt
import random

class MLP_Scratch: 
    def __init__(self, sizes): 
        self.num_layers = len(sizes)
        self.sizes = sizes
        #for each y, np.random.randn(y,1) creates a column vector of y random numbers drawn from a standard normal distribution
        #end up with a list of vectors: each vector is randomly initialized to the number of neurons in the layer
        #skips the first layer as first layer does not have biases
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] 

        #zip(sizes[:-1], sizes[1:]) pairs each layer with the next one.
        #create a matrix with y rows and x columns - initialized randomly
        #Each row of the weight matrix holds all the incoming connection weights for one neuron in the destination alyer
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def sigmoid(self, z): 
        return 1.0/(1.0 + np.exp(-z))
    
    def sigmoid_prime(self, z): 
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def feedforward(self, a): 
        #return the output of the network if the input is a 
        #a is a (n, 1) Numpy ndarray
        for bias, weight in zip(self.biases, self.weights): 
            a = self.sigmoid(np.dot(weight, a) + bias) 
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the data using mini-batch stochastic gradient descent.
        Training data is a list of tuples "(x,y)" representing the training inputs and the desired outputs
        Epochs is the number of passes through the full training data
        If "test_data" is provided then the network will be evaluated using the test data after each epoch
        and partial progress printed - useful for progress but slows things down
        """

        n = len(training_data)
        loss_history = []
        time_history = []
        cumulative_time = 0
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            epoch_start = time.time()
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            cumulative_time += time.time() - epoch_start
            time_history.append(cumulative_time)
            if test_data:
                loss = self.evaluate_mse(test_data)
                loss_history.append(loss)
                print(f"Epoch {j}, Loss: {loss:.4f}")
            else:
                print(f"Epoch {j} complete")
        return loss_history, time_history
    
    def update_mini_batch(self, mini_batch, eta): 
        """
        Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch
        "mini_batch" is a list of tuples  "(x,y)" and "eta" is the learning rate
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch: 
            delta_nabla_b, delta_nabla_w = self.backprop(x,y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y): 
        """
        Return a tuple "(nabla_b, nabla_w)" representing the graident for the cost function C_x.
        "nabla_b" and "nabla_w" are layer-by-layer lists of numpy arrays, similar to "self.biases" and "self.weights"
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x] #list that stores all the activations, layer by layer 
        zs = [] #list to store all z vectors, layer by layer 

        for b, w in zip(self.biases, self.weights): 
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        #backward pass 
        #array[-1] indicates the last element in the array 
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])

        nabla_b[-1] = delta 
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        #Note that l = 1 means the last layer of neurons, l = 2 is the second-last layer and so on. 
        for l in range(2, self.num_layers): 
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp

            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)

    # def evaluate(self, test_data):
    #     test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    #     return sum(int(pred == y) for (pred, y) in test_results)
    
    def evaluate_mse(self, test_data):
        total_loss = 0
        for x, y in test_data: 
            output = self.feedforward(x)
            total_loss += np.mean((output-y) ** 2)
        return total_loss / len(test_data)

    def cost_derivative(self, output_activations, y): 
        """
        Return the vector of partial derivates partial C_x / partial a for the output activations
        """
        return (output_activations-y) #defined for a quadratic cost function 
    
    def print_biases(self): 
        for i in range(len(self.biases)): 
            print(f"biases for neurons in layer {i+1}: \n{self.biases[i]}")
            print(f"shape: {self.biases[i].shape}")

class MLP_Torch(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.layer1 = nn.Linear(16, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 3)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

    def fit(self, X, y, epochs=1000, lr=0.1, X_val=None, y_val=None):
        loss_fn = nn.MSELoss() 
        optimiser = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=50)
        accuracy_history = []
        time_history = []
        cumulative_time = 0
        for epoch in range(epochs):
            self.train()

            epoch_start = time.time()

            logits = self(X) #compute predictions for all training data
            loss = loss_fn(logits, y) #compute loss function between predictions and targets
            optimiser.zero_grad()
            loss.backward() #compute gradient of loss wrt every weight and bias
            optimiser.step() #update weights and biases using adam
            scheduler.step(loss) #lr is halved if loss same for 50 epochs

            cumulative_time += time.time() - epoch_start
            time_history.append(cumulative_time)

            if X_val is not None:
                accuracy_history.append(self.evaluate_float(X_val, y_val))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        return accuracy_history, time_history


    def evaluate_float(self, X, y):  
        #do a forward pass on the test set and return MSE loss
        self.eval()
        with torch.no_grad():
            return nn.MSELoss()(self(X), y).item()

    def evaluate_quantised(self, X, y):  
        #copy model, quantise each parameter to Q4.4, recover quantised float value 
        #run forward pass with quantised weights and biases and return MSE loss
        import copy 
        model_q = copy.deepcopy(self)

        with torch.no_grad(): 
            for param in model_q.parameters(): 
                q = quantise_q4_4(param.data)
                param.copy_(q.float() / 16) 

        model_q.eval()

        with torch.no_grad(): 
            return nn.MSELoss()(model_q(X), y).item()
    
    def calibrate(self, X): 
        self.eval()
        with torch.no_grad(): 
            h1 = torch.relu(self.layer1(X))
            h2 = torch.relu(self.layer2(h1))
            logits = self.layer3(h2)

        z_min = logits.min().item()
        z_max = logits.max().item() 

        z_offset = z_min 
        z_scale = 255.0 / (z_max - z_min)

        z_shift = max(0, int(np.ceil(np.log2(abs(z_max) / 255)))) if abs(z_max) > 255 else 0

        print(f"z_min (float): {z_min:.4f}")
        print(f"z_max (float): {z_max:.4f}")
        print(f"z_scale (float): {z_scale:.4f}")    

        z_offset_q16_16 = int(z_offset * 65536) & 0xFFFFFFFF
        z_scale_q16_16 = int(z_scale * 65536) & 0xFFFFFFFF

        return z_shift, z_offset_q16_16, z_scale_q16_16

    def export(self, path="weights.hex"):
        all_weights = []

        for name, param in self.named_parameters():
            all_weights.extend(quantise_q4_4(param.data).flatten().tolist())

        with open(path, "w") as f:
            for value in all_weights:
                f.write(f"{value & 0xFF:02X}\n")

def quantise_q4_4(tensor):
    scaled = tensor * 16
    clamped = torch.clamp(scaled, -128, 127)
    quantised = torch.round(clamped).to(torch.int8)
    return quantised

iris = load_iris()
X = iris.data.astype(np.float32)
y = iris.target
print(y)

CLASS_COLOURS = np.array([
    [1.0, 0.0, 0.0], #setosa - red
    [0.0, 1.0, 0.0], #versicolor - green
    [0.0, 0.0, 1.0] #virginica - blue
], dtype=np.float32)

y_colours = CLASS_COLOURS[y]

X16 = np.zeros((len(X), 16), dtype=np.float32)
X16[:, :4] = X

X_train, X_test, y_train, y_test = train_test_split(X16, y_colours, test_size=0.2, random_state=42)

#learn min and max from training data, then using those values scale training and test data sets to [0,1]
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#tensors for colour vectors for MLP_Torch
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

#in mlp_scratch training requires a one hot vector, testing requires a list of tuples
_, _, y_train_int, y_test_int = train_test_split(X16, y, test_size=0.2, random_state=42)

def to_one_hot(c, n=3): 
    v = np.zeros((n, 1))
    v[int(c)] = 1.0
    return v

scratch_train = [(X_train[i].reshape(-1,1), to_one_hot(y_train_int[i])) for i in range(len(X_train))]
scratch_test = [(X_test[i].reshape(-1, 1), to_one_hot(y_test_int[i])) for i in range(len(X_test))]

model = MLP_Torch()
print("Torch training...")
torch_loss_history, torch_time_history = model.fit(X_train_torch, y_train_torch, epochs=5000, X_val=X_test_torch, y_val=y_test_torch)
print("Finished Torch Training\n")

loss_float = model.evaluate_float(X_test_torch, y_test_torch)
loss_quantised = model.evaluate_quantised(X_test_torch, y_test_torch)

print(f"Torch MSE loss (float): {loss_float:.4f}")
print(f"Torch MSE loss (quantised): {loss_quantised:.4f}")

model.export("weights.hex")

#compute calibration values
z_shift, z_offset, z_scale = model.calibrate(X_train_torch)
print(f"Calibration values: \n z_shift: {z_shift}\n z_offset: {z_offset}, \n z_scale {z_scale}")



#Scratch Training Code
scratch = MLP_Scratch([16, 64, 32, 3])

print("Scratch Training...")
scratch_start = time.time()
scratch_loss_history, scratch_time_history = scratch.SGD(scratch_train, epochs=500, mini_batch_size=32, eta=3.0, test_data=scratch_test)
scratch_train_time = time.time() - scratch_start
print("Finished Scratch Training...\n")

scratch_mse = scratch.evaluate_mse(scratch_test)

#Comparison Table
print(f"{'='*55}")
print(f"{'':>15} {'Loss':>10} {'Train time':>12}")
print(f"{'MLP_Scratch':>15} {scratch_loss_history[-1]:>10.4f} {scratch_train_time:>11.2f}s  (SGD)")
print(f"{'MLP_Torch':>15} {loss_float:>10.4f} {torch_time_history[-1]:>11.2f}s  (Adam)")
print(f"{'MLP_Torch Q4.4':>15} {loss_quantised:>10.4f}")
print(f"{'='*55}")

#Plots
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(torch_time_history, torch_loss_history, label="MLP_Torch (Adam)")
ax.plot(scratch_time_history, scratch_loss_history, color="orange", label="MLP_Scratch (SGD)")
ax.set_title("Validation Loss vs Training Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("MSE Loss")
ax.legend()

plt.tight_layout()
plt.show()
