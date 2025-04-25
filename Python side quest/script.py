from scipy.special import softmax

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 31
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

class Adam:
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # Initialize first moment vector
        self.v = None  # Initialize second moment vector
        self.t = 0      # Initialize timestep

    def update(self, params, grads):
        if self.m is None:
            self.m = [torch.zeros_like(grad) for grad in grads]
            self.v = [torch.zeros_like(grad) for grad in grads]

        self.t += 1

        updated_params = []
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2

            m_corrected = self.m[i] / (1 - self.beta1**self.t)
            v_corrected = self.v[i] / (1 - self.beta2**self.t)

            param_update = self.learning_rate * m_corrected / (torch.sqrt(v_corrected) + self.epsilon)
            updated_param = param - param_update
            updated_params.append(updated_param)
        return updated_params

class Layer:
    def __init__(self, size, prev_size):
        self.size = size
        self.prev_size = prev_size
        self.lr = 0.01

        self.optimizer = Adam()
        self.input_A_weights = torch.randn(size, prev_size).to(device)
        self.input_B_weights = torch.randn(size, prev_size).to(device)
    
        
        self.table_weights = torch.randn(16, size).to(device)

    def layer_output(self, prev_layer_output: torch.Tensor) -> torch.Tensor:
        self.prev_layer_output = prev_layer_output
        self.table_probs = torch.softmax(self.table_weights, dim=0)
        self.input_A_probs = torch.softmax(self.input_A_weights, dim=1)
        self.input_B_probs = torch.softmax(self.input_B_weights, dim=1)

        a_inputs = torch.matmul(self.input_A_probs, prev_layer_output)
        b_inputs = torch.matmul(self.input_B_probs, prev_layer_output)

        self.table_output = tableOperator(a_inputs, b_inputs)
        self.a_delta = torch.sum(derivativeA(a_inputs, b_inputs) * self.table_probs.unsqueeze(-1), dim=0)
        self.b_delta = torch.sum(derivativeB(a_inputs, b_inputs) * self.table_probs.unsqueeze(-1), dim=0)

        layer_output = torch.sum(self.table_output * self.table_probs.unsqueeze(-1), dim=0)
        return layer_output

    def update_weight_delta(self, error: torch.Tensor):
        table_weight_error = self.table_output * error.unsqueeze(0)
        table_weight_delta = torch.sum(table_weight_error, dim=-1)
        
        self.a_error = self.a_delta * error
        self.b_error = self.b_delta * error

        input_a_weight_delta = torch.matmul(self.a_error, self.prev_layer_output.T)
        input_b_weight_delta = torch.matmul(self.b_error, self.prev_layer_output.T)

        new_parameters = self.optimizer.update(
            [self.table_weights, self.input_A_weights, self.input_B_weights],
            [table_weight_delta, input_a_weight_delta, input_b_weight_delta]
        )
        
        self.table_weights = new_parameters[0]
        self.input_A_weights = new_parameters[1]
        self.input_B_weights = new_parameters[2]


    def prev_layer_error(self, error: torch.Tensor) -> torch.Tensor:
        # Initialize the index count and previous layer error tensors
        prev_layer_error = torch.matmul(self.input_A_probs.T, self.a_error) + torch.matmul(self.input_B_probs.T, self.b_error)
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
network = NeuralNetwork([8, 8, 8, 8, 8, 8, 8, 8, 8, 8])

epochs = 3000
past_error = 10000
for epoch in range(epochs):
    # Forward pass
    output = network.forward(batch_inputs.T)  # Transpose to match input shape
    error = output - batch_targets.T
    
    # Check for convergence (stop if error increases)
    # if torch.sum(error * error) > past_error:
    #     break
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
