from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
import numpy as np 
import random
import time
import matplotlib.pyplot as plt

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

        n_test = len(test_data) if test_data else None
        n = len(training_data)
        accuracy_history = []
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
                acc = self.evaluate(test_data) / n_test
                accuracy_history.append(acc)
                print(f"Epoch {j}: {int(acc * n_test)} / {n_test}")
            else:
                print(f"Epoch {j} complete")
        return accuracy_history, time_history
    
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

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(pred == y) for (pred, y) in test_results)

    def cost_derivative(self, output_activations, y): 
        """
        Return the vector of partial derivates partial C_x / partial a for the output activations
        """
        return (output_activations-y) #defined for a quadratic cost function 
    
    def print_biases(self): 
        for i in range(len(self.biases)): 
            print(f"biases for neurons in layer {i+1}: \n{self.biases[i]}")
            print(f"shape: {self.biases[i].shape}")
        
    
    def print_weights(self): 
        for i in range(len(self.weights)): 
            for j in range(len(self.weights[i])): 
                print(f"weights between layer {i} and layer {i+1}. Neuron {j} in layer {i+1}:\n {self.weights[i]}")
                print(f"shape: {self.weights[i].shape}")

    def test_zip(self): 
        for bias, weight in zip(self.biases, self.weights): 
            print(f"bias:\n {bias} \n weight: \n{weight}\n")

    def print_nabla(self): 
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        print(f"nabla b: {nabla_b}")
        print(f"nabla_w: {nabla_w}")

class MLP_Torch(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.layer1 = nn.Linear(16, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 7)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x) #no activation on the last layer
        return x

    def fit(self, X, y, epochs=10000, lr=0.1, X_val=None, y_val=None):
        #loss_fn = nn.CrossEntropyLoss()
        #optimiser = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=50)
        loss_fn = nn.MSELoss()
        optimiser = torch.optim.SGD(self.parameters(), lr=lr)
        y_onehot = torch.zeros(y.size(0), 7).scatter_(1, y.unsqueeze(1), 1.0)
        accuracy_history = []
        time_history = []
        cumulative_time = 0
        for epoch in range(epochs):
            self.train()
            epoch_start = time.time()
            logits = self(X)
            #loss = loss_fn(logits, y)
            loss = loss_fn(logits, y_onehot)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            #scheduler.step(loss)
            cumulative_time += time.time() - epoch_start
            time_history.append(cumulative_time)
            if X_val is not None:
                accuracy_history.append(self.evaluate_float(X_val, y_val).item())
            if epoch % 1000 == 0:
                #print(f"Epoch {epoch}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        return accuracy_history, time_history

    def evaluate_float(self, X, y):
        self.eval()
        with torch.no_grad():
            predictions = self(X).argmax(dim=1)
            return (predictions == y).float().mean()

    def evaluate_quantised(self, X, y):
        with torch.no_grad():
            for param in self.parameters():
                q = quantise_q4_4(param.data)
                param.copy_(q.float() / 16)
        self.eval()
        with torch.no_grad():
            predictions = self(X).argmax(dim=1)
            return (predictions == y).float().mean()

    def export(self, path="weights.hex"):
        all_weights = []
        for name, param in self.named_parameters():
            all_weights.extend(quantise_q4_4(param.data).flatten().tolist())
        with open(path, "w") as f:
            for value in all_weights:
                f.write(f"{value & 0xFF:02X}\n")

def quantise_q4_4(tensor):
    #Scale float to Q4.4
    scaled = tensor * 16 #mulitply by 2^4 to shift fractional bits
    clamped = torch.clamp(scaled, -128, 127) #clamp to signed range of 8 bits
    quantised = torch.round(clamped).to(torch.int8)
    return quantised

dry_bean = fetch_ucirepo(id=602)

X = dry_bean.data.features
y = dry_bean.data.targets

#20% of data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

le = LabelEncoder()
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

y_train = le.fit_transform(y_train.values.ravel())
y_test = le.transform(y_test.values.ravel())

X_train_np = X_train.copy()
X_test_np  = X_test.copy()
y_train_np = y_train.copy()
y_test_np  = y_test.copy()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

model = MLP_Torch()

print("Torch Training")
start = time.time()
torch_accuracy_history, torch_time_history = model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
torch_train_time = time.time() - start
print("Finished Torch Training\n")


accuracy_float = model.evaluate_float(X_test, y_test)
model.export("weights.hex")
accuracy_quantised = model.evaluate_quantised(X_test, y_test)

end = time.time()

# print(f"Torch Accuracy (quantised Q4.4): {accuracy_quantised:.4f}")
# print(f"Torch Accuracy drop: {(accuracy_float - accuracy_quantised):.4f}")
# print(f"Total values to export: {len(weights)}")
# print(f"Total time taken (Start of Training -> Export Weights): {end - start:.2f}s")

#MLP_Scratch on same data
def to_one_hot(c, n=7):
    v = np.zeros((n, 1))
    v[int(c)] = 1.0
    return v

scratch_train = [(X_train_np[i].reshape(-1, 1), to_one_hot(y_train_np[i])) for i in range(len(X_train_np))]
scratch_test  = [(X_test_np[i].reshape(-1, 1),  int(y_test_np[i]))          for i in range(len(X_test_np))]

scratch = MLP_Scratch([16, 64, 32, 7])

print("Scratch Training")
scratch_start = time.time()
scratch_accuracy_history, scratch_time_history = scratch.SGD(scratch_train, epochs=50, mini_batch_size=32, eta=3.0, test_data=scratch_test)
scratch_train_time = time.time() - scratch_start
print("Finished Scratch Training\n")

scratch_accuracy = scratch_accuracy_history[-1]

print(f"{'='*40}")
print(f"{'':>10} {'Accuracy':>10} {'Train time':>12}")
print(f"{'MLP_Scratch':>10} {scratch_accuracy:>10.4f} {scratch_train_time:>11.2f}s  (50 epochs, SGD)")
print(f"{'MLP_Torch':>10} {accuracy_float.item():>10.4f} {torch_train_time:>11.2f}s  (10000 epochs, Adam)")
print(f"{'='*40}")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(torch_time_history, torch_accuracy_history)
ax1.set_title("MLP_Torch — Accuracy vs Time")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Test Accuracy")
ax1.set_ylim(0, 1)

ax2.plot(scratch_time_history, scratch_accuracy_history, color="orange")
ax2.set_title("MLP_Scratch — Accuracy vs Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Test Accuracy")
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.show()
