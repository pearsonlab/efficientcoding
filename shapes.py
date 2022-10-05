import torch
from torch import nn


class Shape(nn.Module):
    def __init__(self, kernel_size, initial_parameters, num_shapes):
        super().__init__()
        x = torch.arange(kernel_size)
        y = torch.arange(kernel_size)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.kernel_size = kernel_size
        self.register_buffer("grid_x", grid_x.flatten().float())
        self.register_buffer("grid_y", grid_y.flatten().float())

        params = torch.tensor(initial_parameters).unsqueeze(-1).repeat(1, num_shapes)
        self.shape_params = nn.Parameter(params, requires_grad=False)

    def forward(self, kernel_centers, kernel_polarities, normalize=True):
        kernel_x = kernel_centers[:, 0]
        kernel_y = kernel_centers[:, 1]

        dx = kernel_x[None, :] - self.grid_x[:, None]
        dy = kernel_y[None, :] - self.grid_y[:, None]

        W = self.shape_function(dx ** 2 + dy ** 2)
        if normalize:
            W = W / W.norm(dim=0, keepdim=True)

        return W * kernel_polarities

    def shape_function(self, rr):
        raise NotImplementedError


class DifferenceOfGaussianShape(Shape):
    def __init__(self, kernel_size, num_shapes=1):
        super().__init__(kernel_size, [-3, -0.9, 0], num_shapes)

    def shape_function(self, rr):
        logA, logB, logitC = self.shape_params
        a = logA.exp()
        b = logB.exp()
        a = a + b  # make the center smaller than the surround
        max_r = self.kernel_size // 4
        logitlogC = self.shape_params[2]
        logC = - (a - b) * max_r ** 2 * logitlogC.sigmoid()  #to keep it within (0, 1)
        c = logC.exp()
        self.a, self.b, self.c = a.detach(), b.detach(), c.detach()

        if self.shape_params.shape[-1] == 2:
            a = torch.cat([a[:1].repeat(14), a[1:].repeat(6)])
            b = torch.cat([b[:1].repeat(14), b[1:].repeat(6)])
            c = torch.cat([c[:1].repeat(14), c[1:].repeat(6)])
        return torch.exp(-a * rr) - c * torch.exp(-b * rr)


class GaussianShape(Shape):
    def __init__(self, kernel_size, num_shapes=1):
        super().__init__(kernel_size, [-0.75], num_shapes)

    def shape_function(self, rr):
        self.a = self.shape_params.exp()
        return torch.exp(-self.a * rr)


class DifferenceOfTDistributionShape(Shape):
    def __init__(self, kernel_size, num_shapes=1):
        super().__init__(kernel_size, [-3, -0.9, 0], num_shapes)

    def shape_function(self, rr):
        logA, logB, logitlogC = self.shape_params
        a = logA.exp()
        b = logB.exp()
        a = a + b  # make the center smaller than the surround
        max_r = self.kernel_size // 4
        logitlogC = self.shape_params[2]
        logC = - (a - b) * max_r ** 2 * logitlogC.sigmoid()  #to keep it within (0, 1)
        c = logC.exp()
        self.a, self.b, self.c = a.detach(), b.detach(), c.detach()
        nu = 1
        return (1 + a * rr / nu) ** (-(nu + 1) / 2) - c * (1 + b * rr / nu) ** (-(nu + 1) / 2)


class SingleTDistribution(Shape):
    def __init__(self, kernel_size, num_shapes=1):
        super().__init__(kernel_size, [-3], num_shapes)

    def shape_function(self, rr):
        logA = self.shape_params
        a = logA.exp()
        self.a = a.detach()
        nu = 2
        return (1 + a * rr / nu) ** (-(nu + 1) / 2)


def get_shape_module(type):
    return {
        'difference-of-gaussian': DifferenceOfGaussianShape,
        'gaussian': GaussianShape,
        'difference-of-t': DifferenceOfTDistributionShape,
        'single-t': SingleTDistribution,
    }[type]

