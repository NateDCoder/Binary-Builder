import numpy as np
from scipy.special import softmax

np.random.seed(3)

def tableOperator(A, B):
    # Create a table of logical operations using numpy
    operations = np.array([
        0,                                   # 0: False
        A * B,                               # 1: A ∧ B
        A - A * B,                           # 2: ¬(A ⇒ B)
        A,                                   # 3: A
        B - A * B,                           # 4: ¬(A ⇐ B)
        B,                                   # 5: B
        A + B - 2 * A * B,                   # 6: A ⊕ B
        A + B - A * B,                       # 7: A ∨ B
        1 - (A + B - A * B),                 # 8: ¬(A ∨ B)
        1 - (A + B - 2 * A * B),             # 9: ¬(A ⊕ B)
        1 - B,                               # 10: ¬B
        1 - B + A * B,                       # 11: A ⇐ B
        1 - A,                               # 12: ¬A
        1 - A + A * B,                       # 13: A ⇒ B
        1 - A * B,                           # 14: ¬(A ∧ B)
        1                                    # 15: True
    ], dtype=float)

    return operations



def derivativeA(A, B, index):
    derivativesA = [
        lambda A, B: 0,         # 0: False
        lambda A, B: B,         # 1: A ∧ B
        lambda A, B: 1 - B,     # 2: ¬(A ⇒ B)
        lambda A, B: 1,         # 3: A
        lambda A, B: -B,        # 4: ¬(A ⇐ B)
        lambda A, B: 0,         # 5: B
        lambda A, B: 1 - 2 * B, # 6: A ⊕ B
        lambda A, B: 1 - B,     # 7: A ∨ B
        lambda A, B: B - 1,     # 8: ¬(A ∨ B)
        lambda A, B: 2 * B - 1, # 9: ¬(A ⊕ B)
        lambda A, B: 0,         # 10: ¬B
        lambda A, B: B,         # 11: A ⇐ B
        lambda A, B: -1,        # 12: ¬A
        lambda A, B: -1 + B,    # 13: A ⇒ B
        lambda A, B: -B,        # 14: ¬(A ∧ B)
        lambda A, B: 0          # 15: True
    ]
    return derivativesA[index](A, B)

def derivativeB(A, B, index):
    derivativesB = [
        lambda A, B: 0,         # 0: False
        lambda A, B: A,         # 1: A ∧ B
        lambda A, B: -A,        # 2: ¬(A ⇒ B)
        lambda A, B: 0,         # 3: A
        lambda A, B: 1 - A,     # 4: ¬(A ⇐ B)
        lambda A, B: 1,         # 5: B
        lambda A, B: 1 - 2 * A, # 6: A ⊕ B
        lambda A, B: 1 - A,     # 7: A ∨ B
        lambda A, B: A - 1,     # 8: ¬(A ∨ B)
        lambda A, B: 2 * A - 1, # 9: ¬(A ⊕ B)
        lambda A, B: -1,        # 10: ¬B
        lambda A, B: -1 + A,    # 11: A ⇐ B
        lambda A, B: 0,         # 12: ¬A
        lambda A, B: A,         # 13: A ⇒ B
        lambda A, B: -A,        # 14: ¬(A ∧ B)
        lambda A, B: 0          # 15: True
    ]
    return derivativesB[index](A, B)



def dec2Bin(dec):
    bin = ""
    while (dec > 0):
        bin = str(dec % 2) + bin
        dec //= 2
    bin = (8 - len(bin)) * "0" + bin
    return bin

class Nueron: 
    def __init__(self, a_index, b_index):
        self.a_index = a_index
        self.b_index = b_index

        self.weights = np.random.randn(16)

    def output(self, A, B):
        self.probablities = softmax(self.weights)
        self.weight_gradient = tableOperator(A, B) 

        self.a_delta = np.array([derivativeA(A, B, i) for i in range(16)], dtype=float) * self.probablities
        self.b_delta = np.array([derivativeB(A, B, i) for i in range(16)], dtype=float) * self.probablities

        result = np.sum(self.weight_gradient * self.probablities)

        return result
    def update_weights(self, lr):
        self.weights -= self.weight_gradient * lr

class Layer:
    def __init__(self, size, prev_size):
        self.size = size
        self.layer = []
        self.prev_size = prev_size
        for i in range(size):
            indexA = int(np.random.random() * prev_size)
            indexB = int(np.random.random() * (prev_size - 1))

            if (indexA == indexB):
                indexB = prev_size - 1

            self.layer.append(Nueron(indexA, indexB))
    def layer_ouput(self, prev_layer_ouput):
        layer_ouput = np.zeros(self.size)

        for i in range(self.size):
            layer_ouput[i] = self.layer[i].output(prev_layer_ouput[self.layer[i].a_index], prev_layer_ouput[self.layer[i].b_index])
        return layer_ouput
    def update_weight_delta(self, error):
        for i in range(self.size):
            self.layer[i].weight_gradient *= error[i]
          
            self.layer[i].update_weights(0.1)

    def prev_layer_error(self, error):
        index_count = np.zeros(self.prev_size)
        prev_layer_error = np.zeros(self.prev_size)
        for i in range(self.size):
            self.layer[i].a_delta *= error[i]
            self.layer[i].b_delta *= error[i]

            index_count[self.layer[i].a_index] += 1
            index_count[self.layer[i].b_index] += 1


            prev_layer_error[self.layer[i].a_index] += np.sum(self.layer[i].a_delta)
            prev_layer_error[self.layer[i].b_index] += np.sum(self.layer[i].b_delta)

            self.layer[i].update_weights(0.01)
        return prev_layer_error



    
class NueralNetwork:
    def __init__(self, model_size):
        self.layers = []
        for i in range(len(model_size) - 2):
            layer_size = model_size[i+1]
            self.layers.append(Layer(layer_size, model_size[i]))
    def forward(self, input):

        hidden = []

        hidden.append(self.layers[0].layer_ouput(input))
        if (len(self.layers) > 1):
            for i in range(len(self.layers) - 1):
                hidden.append(self.layers[i+1].layer_ouput(hidden[-1]))
        return hidden[-1]
    def backward_prop(self, error):
        # Normalize the initial error (from output layer)
        error = error / (np.linalg.norm(error) + 1e-8)

        # Start from the output layer and propagate backward
        for i in reversed(range(len(self.layers))):
            self.layers[i].update_weight_delta(error)

            # If not the first layer, compute error to propagate to previous layer
            if i > 0:
                error = self.layers[i].prev_layer_error(error)
                error = error / (np.linalg.norm(error) + 1e-8)


# For example, create batch for 0 through 7
batch_inputs = []
batch_targets = []

for i in range(8):
    bin_input = list(dec2Bin(i))
    bin_target = list(dec2Bin((i + 1) % 16))  # wrap-around for 4-bit

    batch_inputs.append([float(x) for x in bin_input])
    batch_targets.append([float(x) for x in bin_target])

batch_inputs = np.array(batch_inputs)   # shape (8, 4)
batch_targets = np.array(batch_targets) # shape (8, 4)

print(batch_inputs)

binary_number = dec2Bin(5)
input = np.array(list(binary_number), dtype=float)

target = np.array(list(dec2Bin(6)), dtype=float)
network = NueralNetwork([8, 8, 8, 8])
epochs = 1000
past_error = 10000
for i in range(epochs):
    output = network.forward(input)
    # output = network.forward(batch_inputs)
    error = output - target
    if (np.sum(error * error) > past_error):
        quit()
    print("Error:", np.sum(error * error))
    past_error = np.sum(error * error)
    network.backward_prop(error)

output = network.forward(input)
error = output - target
print(np.sum(error * error))
print("Network Output:", output)
print("Target Output:", target)
