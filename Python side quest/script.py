from scipy.special import softmax

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 3
torch.manual_seed(seed)

# If using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU

# Optional: Make operations deterministic (slower but fully reproducible)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def tableOperator(A:torch.Tensor, B:torch.Tensor) -> torch.Tensor: 
    result = torch.empty((16,) + A.shape, device=A.device)

    # Perform the logical operations
    result[0]  = 0                                         # 0:False
    result[1]  = A * B                                     # 1:A ∧ B
    result[2]  = A - A * B                                 # 2:¬(A ⇒ B)
    result[3]  = A                                         # 3:A
    result[4]  = B - A * B                                 # 4:¬(A ⇐ B)
    result[5]  = B                                         # 5:B
    result[6]  = A + B - 2 * A * B                         # 6:A ⊕ B
    result[7]  = A + B - A * B                             # 7:A ∨ B
    result[8]  = 1 - (A + B - A * B)                       # 8:¬(A ∨ B)
    result[9]  = 1 - (A + B - 2 * A * B)                   # 9:¬(A ⊕ B)
    result[10] = 1 - B                                     # 10:¬B
    result[11] = 1 - B + A * B                             # 11:A ⇐ B
    result[12] = 1 - A                                     # 12:¬A
    result[13] = 1 - A + A * B                             # 13:A ⇒ B
    result[14] = 1 - A * B                                 # 14:¬(A ∧ B)
    result[15] = 1                                         # 15:True

    return result

def derivativeA(A:torch.Tensor, B:torch.Tensor) -> torch.Tensor: 
    result = torch.empty((16,) + A.shape, device=A.device)

    # Perform derivative calculations for A (corresponding to each operation)
    result[0]  = 0                                         # 0:False
    result[1]  = B                                         # 1:A ∧ B
    result[2]  = 1 - B                                     # 2:¬(A ⇒ B)
    result[3]  = 1                                         # 3:A
    result[4]  = -B                                        # 4:¬(A ⇐ B)
    result[5]  = 0                                         # 5:B
    result[6]  = 1 - 2 * B                                 # 6:A ⊕ B
    result[7]  = 1 - B                                     # 7:A ∨ B
    result[8]  = B - 1                                     # 8:¬(A ∨ B)
    result[9]  = 2 * B - 1                                 # 9:¬(A ⊕ B)
    result[10] = 0                                         # 10:¬B
    result[11] = B                                         # 11:A ⇐ B
    result[12] = -1                                        # 12:¬A
    result[13] = B - 1                                     # 13:A ⇒ B
    result[14] = -B                                        # 14:¬(A ∧ B)
    result[15] = 0                                         # 15:True
    return result

def derivativeB(A:torch.Tensor, B:torch.Tensor) -> torch.Tensor: 
    result = torch.empty((16,) + A.shape, device=A.device)


    # Perform derivative calculations for B (corresponding to each operation)
    result[0]  = 0                                         # 0:False
    result[1]  = A                                         # 1:A ∧ B
    result[2]  = -A                                        # 2:¬(A ⇒ B)
    result[3]  = 0                                         # 3:A
    result[4]  = 1 - A                                     # 4:¬(A ⇐ B)
    result[5]  = 1                                         # 5:B
    result[6]  = 1 - 2 * A                                 # 6:A ⊕ B
    result[7]  = 1 - A                                     # 7:A ∨ B
    result[8]  = A - 1                                     # 8:¬(A ∨ B)
    result[9]  = 2 * A - 1                                 # 9:¬(A ⊕ B)
    result[10] = -1                                        # 10:¬B
    result[11] = A - 1                                     # 11:A ⇐ B
    result[12] = 0                                         # 12:¬A
    result[13] = A                                         # 13:A ⇒ B
    result[14] = -A                                        # 14:¬(A ∧ B)
    result[15] = 0                                         # 15:True
    
    return result
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
        self.weights = torch.randn(16).to(device)

    def output(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Compute the probabilities using the softmax function
        self.probabilities = torch.softmax(self.weights, dim=0).unsqueeze(1)  # Same as [:, np.newaxis] in numpy

        # Compute the weight gradients using the tableOperator function (modified for PyTorch)
        self.weight_gradient = tableOperator(A, B)

        # Compute deltas for A and B using their respective derivatives
        self.a_delta = derivativeA(A, B) * self.probabilities
        self.b_delta = derivativeB(A, B) * self.probabilities

        # Compute the result by summing over the weight gradients, multiplied by the probabilities
        result = torch.sum(self.weight_gradient * self.probabilities, dim=0)

        return result
    def update_weights(self, lr):
        grads = torch.mean(self.weight_gradient, axis=1)

        self.weights -= grads * lr

class Layer:
    def __init__(self, size, prev_size):
        self.size = size
        self.layer = []
        self.prev_size = prev_size
        self.lr = 5
        for i in range(size):
            indexA = torch.randint(0, prev_size, (1,), device=device).item()
            indexB = torch.randint(0, prev_size - 1, (1,), device=device).item()

            if (indexA == indexB):
                indexB = prev_size - 1

            self.layer.append(Nueron(indexA, indexB))
    def layer_output(self, prev_layer_output: torch.Tensor) -> torch.Tensor:
        # Initialize the output tensor
        layer_output = torch.zeros((self.size, prev_layer_output.shape[1]), device=prev_layer_output.device)

        # Loop through each layer and calculate the output
        for i in range(self.size):
            layer_output[i] = self.layer[i].output(prev_layer_output[self.layer[i].a_index], prev_layer_output[self.layer[i].b_index])
        
        return layer_output

    def update_weight_delta(self, error: torch.Tensor):
        for i in range(self.size):
            # Element-wise multiplication for weight gradients and error
            self.layer[i].weight_gradient *= error[i]
            # Update weights with learning rate (e.g., 0.1)
            self.layer[i].update_weights(self.lr)
            self.lr *= .99


    def prev_layer_error(self, error: torch.Tensor) -> torch.Tensor:
        # Initialize the index count and previous layer error tensors
        index_count = torch.zeros(self.prev_size, device=error.device, dtype=torch.int)
        prev_layer_error = torch.zeros((self.prev_size, error.shape[1]), device=error.device)

        # Loop through each layer and calculate the previous layer's error
        for i in range(self.size):
            # Element-wise multiplication for delta values and error
            self.layer[i].a_delta *= error[i]
            self.layer[i].b_delta *= error[i]

            # Update index count and previous layer error
            index_count[self.layer[i].a_index] += 1
            index_count[self.layer[i].b_index] += 1

            prev_layer_error[self.layer[i].a_index] += torch.sum(self.layer[i].a_delta, dim=0)
            prev_layer_error[self.layer[i].b_index] += torch.sum(self.layer[i].b_delta, dim=0)

        return prev_layer_error



    
class NeuralNetwork:
    def __init__(self, model_size):
        self.layers = []
        # Create layers (excluding the input and output layers)
        for i in range(len(model_size) - 2):
            layer_size = model_size[i+1]
            self.layers.append(Layer(layer_size, model_size[i]))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = []
        # First layer output
        hidden.append(self.layers[0].layer_output(input))
        
        # Iterate through the rest of the layers
        if len(self.layers) > 1:
            for i in range(1, len(self.layers)):
                hidden.append(self.layers[i].layer_output(hidden[-1]))
        
        return hidden[-1]

    def backward_prop(self, error: torch.Tensor):
        # Normalize the initial error (from the output layer)
        error = error / (torch.norm(error) + 1e-8)
        
        # Start from the output layer and propagate backward
        for i in reversed(range(len(self.layers))):
            # Update the weight deltas for the current layer
            self.layers[i].update_weight_delta(error)
            
            # If not the first layer, compute the error to propagate to the previous layer
            if i > 0:
                error = self.layers[i].prev_layer_error(error)
                error = error / (torch.norm(error) + 1e-8)


# Create batch for 0 through 7
batch_inputs = []
batch_targets = []

for i in range(256):
    bin_input = list(dec2Bin(i))
    bin_target = list(dec2Bin((i + 1) % 16))  # wrap-around for 4-bit

    batch_inputs.append([float(x) for x in bin_input])
    batch_targets.append([float(x) for x in bin_target])

# Convert lists to PyTorch tensors
batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32).to(device)  # shape (8, 4)
batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(device) # shape (8, 4)

# Initialize Neural Network (assuming you've implemented the class as before)
network = NeuralNetwork([8, 64, 64, 644, 64, 64, 64, 8, 8])

epochs = 1000
past_error = 10000

for epoch in range(epochs):
    # Forward pass
    output = network.forward(batch_inputs.T)  # Transpose to match input shape
    error = output - batch_targets.T
    
    # Check for convergence (stop if error increases)
    if torch.sum(error * error) > past_error:
        break
    print("Error:", torch.sum(error * error).item())  # .item() to get the Python scalar value
    past_error = torch.sum(error * error).item()
    
    # Backward pass (update weights)
    network.backward_prop(error)
print(batch_inputs.device)
# Final output and error calculation
output = network.forward(batch_inputs.T)
error = output - batch_targets.T
print(torch.sum(error * error).item())
print("Network Input:", batch_inputs)
print("Network Output:", output.T)
print("Target Output:", batch_targets)
