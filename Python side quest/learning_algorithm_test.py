import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

    
class AvoidingMinimum(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([4.0]))
        # self.logged_grads = [] 
    def forward(self, input:torch.Tensor):
        output = self.a * input *  torch.cos(self.a * input)
        def new_gradient(grad):
            print(torch.sum(grad))
            
            print(self.a.data)
            self.a_gradient = torch.sign(torch.sum(grad)) * torch.sum(torch.abs(grad))
         
            return grad
        if output.requires_grad:
            output.register_hook(new_gradient)
        return output
    

x = np.linspace(-np.pi+np.pi/100, np.pi, 201)
y = np.sin(10 * x)

x = torch.tensor(x).to(device)
y = torch.tensor(y).to(device)


model = nn.Sequential(
    AvoidingMinimum()
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=[0.95, 0.999])
criterion = nn.L1Loss()
training_x = []
training_y = []
epochs = 200

for epoch in range(epochs):
    training_x.append(float(model[0].a.data))
    optimizer.zero_grad()
    
    output = model(x)
    loss = criterion(output, y)

    loss.backward()
    
    gradients = model[0].a.grad.detach().clone()
    optimizer.step()
    
    # error_over_time.append(loss.item())
    print(f"Epoch {epoch}: Loss = {loss.item():.6f}: Gradients = {gradients}")
    
    training_y.append(loss.item())

parameter_values = np.linspace(-4, 4, 1000)
errors = []
with torch.no_grad():
    for parameter in parameter_values:
        model[0].a.data = torch.tensor([parameter], device=device)
        output = model(x)
        loss = criterion(output, y).item()
        errors.append(loss)
        
fig, ax = plt.subplots()

plt.plot(parameter_values, errors)
colors = np.array(range(len(training_x))) / len(training_x)
plt.scatter(training_x, training_y, s=5, c=colors, cmap='viridis')

plt.show()