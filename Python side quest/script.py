from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 3141
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

class LogicLayer(nn.Module):
    def __init__(self, prev_size, size):
        super().__init__()
        self.size = size
        self.prev_size = prev_size

        self.input_A_weights = nn.Parameter(torch.randn(size, prev_size))
        self.input_B_weights = nn.Parameter(torch.randn(size, prev_size))
        self.table_weights = nn.Parameter(torch.randn(16, size))  # 16 logic ops per neuron

    def forward(self, prev_layer_output):
        prev_layer_output = GradFactor.apply(prev_layer_output, 10)
        # Softmax for differentiable input selectors
        input_A_probs = F.softmax(self.input_A_weights, dim=1)
        input_B_probs = F.softmax(self.input_B_weights, dim=1)
        table_probs = F.softmax(self.table_weights, dim=0)

        a_inputs = GradFactor.apply(torch.matmul(input_A_probs, prev_layer_output), 10)
        b_inputs = GradFactor.apply(torch.matmul(input_B_probs, prev_layer_output), 10)

        table_output = tableOperator(a_inputs, b_inputs)  # shape: [16, batch_size, size]
        output = torch.sum(table_output * table_probs.unsqueeze(-1), dim=0)  # shape: [batch_size, size]

        return output

    
# Create batch for 0 through 7
batch_inputs = []
batch_targets = []

for i in range(200):
    bin_input = list(dec2Bin(i))
    bin_target = list(dec2Bin((i + 1) % 256)) 

    batch_inputs.append([float(x) for x in bin_input])
    batch_targets.append([float(x) for x in bin_target])

# Convert lists to PyTorch tensors
batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32).to(device) 
batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(device) 

model = torch.nn.Sequential(
    LogicLayer(8, 64),
    LogicLayer(64, 64),
    LogicLayer(64, 64),
    LogicLayer(64, 64),
    LogicLayer(64, 64),
    LogicLayer(64, 64),
    LogicLayer(64, 8),
).to(device)


# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=[0.99, 0.999])
criterion = nn.MSELoss()

error_over_time = []
# Training loop
epochs = 15000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    output = model(batch_inputs.T)
    loss = criterion(output, batch_targets.T)

    loss.backward()
    optimizer.step()
    error_over_time.append(loss.item())
    print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
batch_inputs = []
batch_targets = []
   
for i in range(255):
    bin_input = list(dec2Bin(i))
    bin_target = list(dec2Bin((i + 1) % 256)) 

    batch_inputs.append([float(x) for x in bin_input])
    batch_targets.append([float(x) for x in bin_target])
batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32).to(device) 
batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(device) 
    
# Final output
with torch.no_grad():
    output = model(batch_inputs.T)
    final_loss = criterion(output, batch_targets.T).item()
    
torch.set_printoptions(precision=2, sci_mode=False)
print(f"\nFinal Loss: {final_loss:.6f}")
print(batch_targets[-5:])
print("Sample Output:\n", torch.round(output.T[-5:]))


name, param = list(model.named_parameters())[0]
print(f"Layer: {name}, Weights: { F.softmax(param.data, dim=1)}")
name, param = list(model.named_parameters())[1]
print(f"Layer: {name}, Weights: { F.softmax(param.data, dim=1)}")
# for name, param in model.named_parameters():
#     print(f"Layer: {name}, Weights: { (param.data)}")

x_values = range(len(error_over_time))

fig, ax = plt.subplots()

ax.plot(x_values, error_over_time)
plt.show()
