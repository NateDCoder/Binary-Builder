import torch
import difflogic_cuda
import numpy as np
from .functional import bin_op_s, get_unique_connections, GradFactor


class LogicLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, grad_factor=1., connections='random'):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(out_dim, 16, device='cuda'))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grad_factor = grad_factor

        # Create input connections (a, b)
        self.indices = self.get_connections(connections)

        # Precompute indices for CUDA backward efficiency
        a_np, b_np = self.indices[0].cpu().numpy(), self.indices[1].cpu().numpy()
        given_x_indices_of_y = [[] for _ in range(in_dim)]
        for y in range(out_dim):
            given_x_indices_of_y[a_np[y]].append(y)
            given_x_indices_of_y[b_np[y]].append(y)

        self.given_x_indices_of_y_start = torch.tensor(
            np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device='cuda', dtype=torch.int64
        )
        self.given_x_indices_of_y = torch.tensor(
            [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device='cuda'
        )

    def forward(self, x):
        x = GradFactor.apply(x, self.grad_factor)
        x = x.transpose(0, 1).contiguous()
        w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
        return LogicLayerCudaFunction.apply(
            x, self.indices[0], self.indices[1], w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
        ).transpose(0, 1)

    def get_connections(self, connections):
        c = torch.randperm(2 * self.out_dim) % self.in_dim
        c = c.reshape(2, self.out_dim)
        return c[0].to(torch.int64).cuda(), c[1].to(torch.int64).cuda()
class GroupSum(torch.nn.Module):
    def __init__(self, k, tau=1.):
        super().__init__()
        self.k = k
        self.tau = tau

    def forward(self, x):
        return x.view(*x.shape[:-1], self.k, -1).sum(-1) / self.tau
class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, gx_start, gx_indices):
        ctx.save_for_backward(x, a, b, w, gx_start, gx_indices)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, w, gx_start, gx_indices = ctx.saved_tensors
        grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y.contiguous(), gx_start, gx_indices)
        grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        return grad_x, None, None, grad_w, None, None