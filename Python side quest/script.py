from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 315
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

class GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None
    
def apply_weight_dropout(weights, dropout_rate):
    if dropout_rate == 0:
        return weights
    
    mask = (torch.rand_like(weights) > dropout_rate)
    dropped_weights = weights.masked_fill(mask == 0, float('-inf'))
    return dropped_weights
torch.set_printoptions(precision=6, sci_mode=False)
class LogicLayer(nn.Module):
    def __init__(self, prev_size, size):
        super().__init__()
        self.size = size
        self.prev_size = prev_size

        self.input_A_weights = nn.Parameter(torch.randn(size, prev_size) * 0.05)
        self.input_B_weights = nn.Parameter(torch.randn(size, prev_size) * 0.05)
        self.table_weights = nn.Parameter(torch.randn(16, size)  * 0.05)  # 16 logic ops per neuron

    def forward(self, prev_layer_output):
        if not torch.is_grad_enabled():
            a_indices = torch.argmax(self.input_A_weights, dim=1)
            b_indices = torch.argmax(self.input_B_weights, dim=1)
            
            table_indices = torch.argmax(self.table_weights, dim=0)
  
            input_A_probs = torch.nn.functional.one_hot(a_indices, num_classes=self.input_A_weights.size(1)).float()
            input_B_probs = torch.nn.functional.one_hot(b_indices, num_classes=self.input_B_weights.size(1)).float()
            table_probs = torch.nn.functional.one_hot(table_indices, num_classes=self.table_weights.size(0)).float().T
            
            a_inputs = torch.matmul(input_A_probs, prev_layer_output)
            b_inputs = torch.matmul(input_B_probs, prev_layer_output)
            
            table_output = tableOperator(a_inputs, b_inputs)  # shape: [16, batch_size, size]
            output = torch.sum(table_output * table_probs.unsqueeze(-1), dim=0)  # shape: [batch_size, size]
            
            return output
            
        prev_layer_output = GradFactor.apply(prev_layer_output, 1)
        # Softmax for differentiable input selectors
        input_A_probs = F.softmax(self.input_A_weights, dim=1)
        input_B_probs = F.softmax(self.input_B_weights, dim=1)
        table_probs = F.softmax(self.table_weights, dim=0)

        a_inputs = GradFactor.apply(torch.matmul(input_A_probs, prev_layer_output), 1)
        b_inputs = GradFactor.apply(torch.matmul(input_B_probs, prev_layer_output), 1)

        table_output = tableOperator(a_inputs, b_inputs)  # shape: [16, batch_size, size]
        output = torch.sum(table_output * table_probs.unsqueeze(-1), dim=0)  # shape: [batch_size, size]

        try:
            def hook(grad):
                return grad
            output.register_hook(hook)
        except:
            pass
        return output
class BinaryToDecimalMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Make the weights once: [2^7, 2^6, ..., 2^0]
        self.register_buffer("weights", 1.3 ** torch.arange(7, -1, -1).float().to(device).unsqueeze(-1))

    def forward(self, output, target):
        # output, target shape: [batch_size, 8]
        # Compute decimal values
        # print(output.shape, target.shape, self.weights.shape)
        # print(output.T[0], target.T[0], self.weights)
        error_rate = 1
        output_decimal = (output * error_rate)
        target_decimal = (target * error_rate)
        return F.mse_loss(output_decimal, target_decimal)
    
# Create batch for 0 through 7
batch_inputs = []
batch_targets = []
# for _ in range(10):
for i in range(250):
    bin_input = list(dec2Bin(i))
    bin_target = list(dec2Bin((i+1) % 256)) 

    batch_inputs.append([float(x) for x in bin_input])
    batch_targets.append([float(x) for x in bin_target])
for i in range(5):
    i+=2
    bin_input = list(dec2Bin(2**i - 1))
    bin_target = list(dec2Bin(2**i))
    
    batch_inputs.append([float(x) for x in bin_input])
    batch_targets.append([float(x) for x in bin_target])

# Convert lists to PyTorch tensors
batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32).to(device) 
batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(device) 

model = torch.nn.Sequential(
    LogicLayer(8, 8)
).to(device)


# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=[0.99, 0.999])
criterion = BinaryToDecimalMSELoss()

error_over_time = []
# Training loop
epochs = 2000


batch_size = len(batch_inputs) // 10  # for 5 batches

output_grads = []

tableOperatorOverTime = []
for i in range(8):
    tableOperatorOverTime.append([])
    
for epoch in range(epochs):
    epoch += 1
    # Shuffle indices
    indices = torch.randperm(len(batch_inputs))
    inputs_shuffled = batch_inputs[indices]
    targets_shuffled = batch_targets[indices]
    # After 100 epochs, add another LogicLayer
    if epoch % 30 == 0 and epoch < 240:
        print("Appending new LogicLayer...")
        # Detach old model and wrap in new Sequential
        model = torch.nn.Sequential(
            model,  # existing layers
            LogicLayer(8, 8).to(device)  # new layer
        ).to(device)
        
        # Reset optimizer for new parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=[0.99, 0.999])
    
    batch_losses = []
    for i in range(0, len(batch_inputs), batch_size):
        batch_input = inputs_shuffled[i:i+batch_size]
        batch_target = targets_shuffled[i:i+batch_size]

        optimizer.zero_grad()

        # Forward pass with hooks
        output_grads.clear()

        output = model(batch_input.T)
        loss = criterion(output, batch_target.T)
        loss.backward()
        optimizer.step()

        error_over_time.append(loss.item())
        
        batch_losses.append((i, loss.item()))
        
        try:
            _, param = list(model.named_parameters())[1]
            param = F.softmax(param.data, dim=1)
            for i in range(len(param[0])):
                tableOperatorOverTime[i].append(float(param[3][i]))
        except:
           for i in range(8):
                tableOperatorOverTime[i].append(0) 
    top_n = 1
    worst_batches_input = batch_inputs[-5:]
    worst_batches_targets = batch_targets[-5:]

    print(f"Re-training top {top_n} worst batches...")
    for _ in range(10):
        optimizer.zero_grad()
        output = model(worst_batches_input.T)
        loss = criterion(output, worst_batches_targets.T)
        loss.backward()
        optimizer.step()

        error_over_time.append(loss.item())
    print(f"Epoch {epoch+1}: Last Batch Loss = {loss.item():.6f}")

# print(output_grads)
batch_inputs = []
batch_targets = []
   
for i in range(255):
    bin_input = list(dec2Bin(i))
    bin_target = list(dec2Bin((i+1) % 256)) 

    batch_inputs.append([float(x) for x in bin_input])
    batch_targets.append([float(x) for x in bin_target])
batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32).to(device) 
batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(device) 
model.training = False

# Final output
with torch.no_grad():
    output = model(batch_inputs.T)
    final_loss = criterion(output, batch_targets.T).item()
print(f"\nFinal Loss: {final_loss:.6f}")
print(batch_targets[:5])
print("Sample Output:\n", output.T[:5])


# name, param = list(model.named_parameters())[0]
# print(f"Layer: {name}, Weights: { F.softmax(param.data, dim=1)}")
# name, param = list(model.named_parameters())[1]
# print(f"Layer: {name}, Weights: { F.softmax(param.data, dim=1)}")
# name, param = list(model.named_parameters())[2]
# print(f"Layer: {name}, Weights: { F.softmax(param.data, dim=0)}")
torch.set_printoptions(precision=2, sci_mode=False)
i = 0
for name, param in model.named_parameters():
    if "input_A_weights" in name:
        print(f"const INPUT_A_PROBS_{math.floor(i/3)} = { F.softmax(param.data, dim=1)}")
    elif "input_B_weights" in name:
        print(f"const INPUT_B_PROBS_{math.floor(i/3)} = { F.softmax(param.data, dim=1)}")
    elif "table" in name:
        print(f"const TABLE_PROBS_{math.floor(i/3)} = { F.softmax(param.data, dim=0)}")
    i += 1
x_values = range(len(error_over_time))

fig, ax = plt.subplots()

ax.plot(x_values, error_over_time)
for i, value in enumerate(tableOperatorOverTime):
    ax.plot(range(len(value)), value, label=f'Line {i}')

ax.legend()
plt.show()
