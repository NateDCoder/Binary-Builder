from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import math

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 314 #0.048120
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
    AB = A * B
    A_or_B = A + B - AB
    A_xor_B = A + B - 2 * AB
    
    # Perform the logical operations
    result[0]  = 0                                         # 0:False
    result[1]  = AB                                        # 1:A ∧ B
    result[2]  = A - AB                                    # 2:¬(A ⇒ B)
    result[3]  = A                                         # 3:A
    result[4]  = B - AB                                    # 4:¬(A ⇐ B)
    result[5]  = B                                         # 5:B
    result[6]  = A_xor_B                                   # 6:A ⊕ B
    result[7]  = A_or_B                                    # 7:A ∨ B
    result[8]  = 1 - A_or_B                                # 8:¬(A ∨ B)
    result[9]  = 1 - A_xor_B                               # 9:¬(A ⊕ B)
    result[10] = 1 - B                                     # 10:¬B
    result[11] = 1 - B + AB                                # 11:A ⇐ B
    result[12] = 1 - A                                     # 12:¬A
    result[13] = 1 - A + AB                                # 13:A ⇒ B
    result[14] = 1 - AB                                    # 14:¬(A ∧ B)
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
        new_grad = grad_y / grad_y.norm() + 1e-8
        return new_grad, None
    
def apply_probs_dropout(probs, dropout_rate):
    if dropout_rate == 0:
        return probs

    num_outputs = probs.size(1)  # number of neurons (columns)
    device = probs.device

    # Sample dropout mask: True → use one-hot, False → use soft probs
    dropout_mask = torch.rand(num_outputs, device=device) < dropout_rate  # [8]

    # Sample indices from categorical distributions for all neurons
    dist = torch.distributions.Categorical(probs.T)
    sampled_indices = dist.sample()  # shape: [8]

    # Create one-hot vectors
    one_hot = F.one_hot(sampled_indices, num_classes=probs.size(0)).float().T  # shape: [16, 8]

    # Broadcast mask and combine
    dropout_mask = dropout_mask.unsqueeze(0).float()  # shape: [1, 8]
    final_probs = dropout_mask * one_hot + (1 - dropout_mask) * probs  # shape: [16, 8]

    return final_probs
def apply_connection_dropout(probs, dropout_rate):
    if dropout_rate == 0:
        return probs

    num_outputs = probs.size(1)  # number of neurons (columns)
    device = probs.device

    # Sample dropout mask: True → use one-hot, False → use soft probs
    dropout_mask = torch.rand(num_outputs, device=device) < dropout_rate  # [8]

    # Sample indices from categorical distributions for all neurons
    dist = torch.distributions.Categorical(probs)
    sampled_indices = dist.sample()

    # Create one-hot vectors
    one_hot = F.one_hot(sampled_indices, num_classes=probs.size(1)) # shape: [16, 8]
    # Broadcast mask and combine
    dropout_mask = dropout_mask.unsqueeze(-1).float()  # shape: [1, 8]
   
    final_probs = dropout_mask * one_hot + (1 - dropout_mask) * probs  # shape: [16, 8]

    return final_probs

torch.set_printoptions(precision=6, sci_mode=False)
drop_out_rate = 0.0

class LogicLayer(nn.Module):
    def __init__(self, prev_size, size, index = -1):
        super().__init__()
        self.size = size
        self.prev_size = prev_size

        self.input_A_weights = nn.Parameter(torch.randn(size, prev_size)) 
        self.input_B_weights = nn.Parameter(torch.randn(size, prev_size))
  
        
        self.table_weights = nn.Parameter(torch.randn(16, size))  # 16 logic ops per neuron
        self.index = index
        
    def forward(self, prev_layer_output):
        global drop_out_rate
        # if not torch.is_grad_enabled():
        #     a_indices = torch.argmax(self.input_A_weights, dim=1)
        #     b_indices = torch.argmax(self.input_B_weights, dim=1)
            
        #     table_indices = torch.argmax(self.table_weights, dim=0)
  
        #     input_A_probs = torch.nn.functional.one_hot(a_indices, num_classes=self.input_A_weights.size(1)).float()
        #     input_B_probs = torch.nn.functional.one_hot(b_indices, num_classes=self.input_B_weights.size(1)).float()
        #     table_probs = torch.nn.functional.one_hot(table_indices, num_classes=self.table_weights.size(0)).float().T
            
        #     a_inputs = torch.matmul(input_A_probs, prev_layer_output)
        #     b_inputs = torch.matmul(input_B_probs, prev_layer_output)
            
        #     table_output = tableOperator(a_inputs, b_inputs)  # shape: [16, batch_size, size]
        #     output = torch.sum(table_output * table_probs.unsqueeze(-1), dim=0)  # shape: [batch_size, size]
            
        #     return output
            
        prev_layer_output = GradFactor.apply(prev_layer_output, 2.5)
        input_A_weights = self.input_A_weights
        input_B_weights = self.input_B_weights
        
        # Softmax for differentiable input selectors
        input_A_probs = F.softmax(input_A_weights, dim=1)
        input_B_probs = F.softmax(input_B_weights, dim=1)
        table_probs = apply_probs_dropout(F.softmax(self.table_weights, dim=0), drop_out_rate)
        
        def forward_multiplier(input_A_weights, input_B_weights, table_weights):
            input_A_probs = F.softmax(input_A_weights, dim=1)
            input_B_probs = F.softmax(input_B_weights, dim=1)

            table_probs = F.softmax(table_weights, dim=0)  # ignore dropout for deriv
            a_Zeroes = (1 - (table_probs[0] + table_probs[5] + table_probs[10] + table_probs[15])).unsqueeze(-1)
            b_Zeroes = (1 - (table_probs[0] + table_probs[3] + table_probs[12] + table_probs[15])).unsqueeze(-1)

            filter_values = input_A_probs * a_Zeroes + input_B_probs * b_Zeroes
            summed = torch.sum(filter_values, dim=0)
            clamped = torch.clamp(summed, 1e-8, 1)
            multiplier = 1 / clamped - 1
            return multiplier

  
        table_probs = apply_probs_dropout(F.softmax(self.table_weights, dim=0), drop_out_rate)
   
        input_A_probs = F.softmax(input_A_weights, dim=1)
        input_B_probs = F.softmax(input_B_weights, dim=1)
        if torch.is_grad_enabled():
            multiplier = forward_multiplier(input_A_weights, input_B_weights, self.table_weights)
            
            # grads = torch.autograd.grad(
            #     outputs=multiplier.sum(),
            #     inputs=[input_A_weights, input_B_weights],
            #     create_graph=True  # keep graph for higher-order grads
            # )
            
            self.multiplier = multiplier
   
            input_A_probs = apply_connection_dropout(F.softmax(input_A_weights + multiplier, dim=1), 0)
            input_B_probs = apply_connection_dropout(F.softmax(input_B_weights + multiplier, dim=1), 0)
         
        # if not torch.is_grad_enabled():
        #     print("old multiplier", multiplier)
        #     a_Zeroes = (1 - (table_probs[0] + table_probs[5] + table_probs[10] + table_probs[15])).unsqueeze(-1)
        #     b_Zeroes = (1 - (table_probs[0] + table_probs[3] + table_probs[12] + table_probs[15])).unsqueeze(-1)
        
        #     multiplier = 1 / torch.clamp(torch.sum(input_A_probs * a_Zeroes + input_B_probs * b_Zeroes, dim=0), 0, 1) - 1
        #     print("new multiplier", multiplier)
        
        a_inputs = GradFactor.apply(torch.matmul(input_A_probs, prev_layer_output), 2.5)
        b_inputs = GradFactor.apply(torch.matmul(input_B_probs, prev_layer_output), 2.5)

        table_output = tableOperator(a_inputs, b_inputs)  # shape: [16, batch_size, size]
        output = GradFactor.apply(torch.sum(table_output * table_probs.unsqueeze(-1), dim=0), 2.5)  # shape: [batch_size, size]

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
# for _ in range(100):
for i in range(255):
    bin_input = list(dec2Bin(i))
    bin_target = list(dec2Bin((i) % 256)) 

    batch_inputs.append([float(x) for x in bin_input])
    batch_targets.append([float(x) for x in bin_target])
# for i in range(5):
#     i+=2
#     bin_input = list(dec2Bin(2**i - 1))
#     bin_target = list(dec2Bin(2**i))
    
#     batch_inputs.append([float(x) for x in bin_input])
#     batch_targets.append([float(x) for x in bin_target])

# Convert lists to PyTorch tensors
batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32).to(device) 
batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(device) 

model = torch.nn.Sequential(
    LogicLayer(8, 8, 0),
    LogicLayer(8, 8),
    LogicLayer(8, 8),
    LogicLayer(8, 8)
).to(device)


# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=[0.99, 0.999])

# def apply_entropy(param, ):
#     def hook(grad, p=param):
#         if p.shape[0] == 16:
#             probs = torch.softmax(param,  dim=0)
#         else:
#             probs = torch.softmax(param,  dim=1)
#         entropy = 1 - torch.mean(probs * torch.log(probs)) * grad.shape[0] * 0.00
#         return grad * entropy
#     param.register_hook(hook)
    
# for param in model.parameters():
#     if param.requires_grad:
#         apply_entropy(param)

criterion = BinaryToDecimalMSELoss()

error_over_time = []
# Training loop
epochs = 2000


batch_size = len(batch_inputs) // 5  # for 5 batches

output_grads = []

tableOperatorOverTime = []
for i in range(8):
    tableOperatorOverTime.append([])
    
for epoch in range(epochs):
    epoch += 1
    # if epoch % 1000 < 500 and epoch > 2000:
    #     drop_out_rate = 0.15
    # else:
    #     drop_out_rate = 0
    # Shuffle indices     
    indices = torch.randperm(len(batch_inputs))
    inputs_shuffled = batch_inputs[indices]
    targets_shuffled = batch_targets[indices]
    # After 100 epochs, add another LogicLayer
    # if epoch % 50 == 0 and epoch < 100:
    #     print("Appending new LogicLayer...")
    #     # Detach old model and wrap in new Sequential
    #     model = torch.nn.Sequential(
    #         model,  # existing layers
    #         LogicLayer(4, 4).to(device)  # new layer
    #     ).to(device)
        
    #     # Reset optimizer for new parameters
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=[0.99, 0.999])
    
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
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, LogicLayer) and layer.multiplier is not None:
                    # Apply multiplier to weights
                    # Broadcast: multiplier shape must match input_dim
                    m = layer.multiplier 
                    layer.input_A_weights.data = torch.clamp(layer.input_A_weights.data + m, -1, 5)
                    layer.input_B_weights.data = torch.clamp(layer.input_B_weights.data + m, -1, 5)
                    
                    # layer.table_weights.data = torch.clamp(layer.table_weights.data, -1, 5)
        optimizer.step()

        error_over_time.append(loss.item())
        
        batch_losses.append((i, loss.item()))
        
        try:
            # _, param1 = list(model.named_parameters())[0]
            # _, param2 = list(model.named_parameters())[1]
            # param1 = F.softmax(param1.data, dim=1)
            # param2 = F.softmax(param2.data, dim=1)
            # for i in range(len(param1)):
            #     tableOperatorOverTime[i].append(float(param1[i][4]) + float(param2[i][4]))
            for i in range(len(model[0].multiplier)):
                tableOperatorOverTime[i].append(float(model[0].multiplier[i]))
        except:
            print("Crashing out")
            for i in range(8):
                tableOperatorOverTime[i].append(0) 
    # top_n = 1
    # worst_batches_input = batch_inputs[-5:]
    # worst_batches_targets = batch_targets[-5:]

    # print(f"Re-training top {top_n} worst batches...")
    # for _ in range(10):
    #     optimizer.zero_grad()
    #     output = model(worst_batches_input.T)
    #     loss = criterion(output, worst_batches_targets.T)
    #     loss.backward()
    #     optimizer.step()

    #     error_over_time.append(loss.item())
    print(f"Epoch {epoch+1}: Last Batch Loss = {loss.item():.6f}")


# print(output_grads)
# batch_inputs = []
# batch_targets = []
   
# for i in range(15):
#     bin_input = list(dec2Bin(i))
#     bin_target = list(dec2Bin((i+1) % 256)) 

#     batch_inputs.append([float(x) for x in bin_input])
#     batch_targets.append([float(x) for x in bin_target])
# batch_inputs = torch.tensor(batch_inputs, dtype=torch.float32).to(device) 
# batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(device) 
# model.training = False

# Final output
with torch.no_grad():
    output = model(batch_inputs.T)
    final_loss = criterion(output, batch_targets.T).item()

output = model(batch_inputs.T)
torch.set_printoptions(precision=2, sci_mode=False)
print(f"\nFinal Loss: {final_loss:.6f}")
print(batch_targets[:5])
print("Sample Output:\n", output.T)

# name, param = list(model.named_parameters())[0]
# print(f"Layer: {name}, Weights: { F.softmax(param.data, dim=1)}")
# name, param = list(model.named_parameters())[1]
# print(f"Layer: {name}, Weights: { F.softmax(param.data, dim=1)}")
# name, param = list(model.named_parameters())[2]
# print(f"Layer: {name}, Weights: { F.softmax(param.data, dim=0)}")

original = []
direction1 = []
direction2 = []

i = 0
for name, param in model.named_parameters():
    original.append(param.data)
    direction1.append(torch.rand_like(param.data) * 5)
    direction2.append(torch.rand_like(param.data) * 5)
    if "input_A_weights" in name:
        print(f"const INPUT_A_PROBS_{math.floor(i/3)} = { param.data.cpu().tolist()}")
    elif "input_B_weights" in name:
        print(f"const INPUT_B_PROBS_{math.floor(i/3)} = { param.data.cpu().tolist()}")
    elif "table" in name:
        print(f"const TABLE_PROBS_{math.floor(i/3)} = { param.data.cpu().tolist() }")
    i += 1

X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = X[i, j]
        y = Y[i, j]
        
        # Modify model parameters
        with torch.no_grad():
            for k, (name, param) in enumerate(model.named_parameters()):
                param.data = (original[k] + x * direction1[k] + y * direction2[k])
        
        # Compute loss
        output = model(batch_inputs.T)
        loss = criterion(output, batch_targets.T)
        Z[i, j] = loss.item()
        if math.isnan(loss.item()):
            quit()
fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax2.set_zlim(np.min(Z), np.max(Z))
ax2.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax2.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig2.colorbar(surf, shrink=0.5, aspect=5)

x_values = range(len(error_over_time))

fig, ax = plt.subplots()

ax.plot(x_values, error_over_time)
for i, value in enumerate(tableOperatorOverTime):
    ax.plot(range(len(value)), value, label=f'Line {i}')

ax.legend()
plt.show()
