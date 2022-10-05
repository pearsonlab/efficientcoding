import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from shapes import get_shape_module, Shape


class DiffExponentialShape(nn.Module):
    def __init__(self,
                 neurons: int,
                 frames: int,
                 individual_shapes: bool = True):
        super().__init__()
        self.max_tau = (frames - 1) / 3.0

        if not individual_shapes:
            self.scale1 = nn.Parameter(0.02 * torch.randn(1, 1, 1) + 1.1)
            self.scale2 = nn.Parameter(0.02 * torch.randn(1, 1, 1) - 0.95)
            self.logit_tau1 = nn.Parameter(0.02 * torch.randn(1, 1, 1) + 1.1)
            self.logit_tau2 = nn.Parameter(0.02 * torch.randn(1, 1, 1) + 1.1 * 1.25)

        else:
            self.scale1 = nn.Parameter(0.02 * torch.randn(neurons, 1, 1) + 1.1)
            self.scale2 = nn.Parameter(0.02 * torch.randn(neurons, 1, 1) - 0.95)
            self.logit_tau1 = nn.Parameter(0.02 * torch.randn(neurons, 1, 1) + 1.1)
            self.logit_tau2 = nn.Parameter(0.02 * torch.randn(neurons, 1, 1) + 1.1 * 1.25)

        self.nf1 = 6
        self.nf2 = 6
        self.frames = frames
        self.individual_shapes = individual_shapes

    def forward(self):
        timepts = torch.arange(self.frames, device=self.scale1.device)  # shape = [K] --> 1, 1, K
        tau1 = self.max_tau * self.logit_tau1.sigmoid()  # shape = [J, 1, 1]
        tau2 = self.max_tau * self.logit_tau2.sigmoid()
        t1 = timepts / tau1  # shape = [J, 1, K]
        t2 = timepts / tau2
        filter1 = self.scale1 * t1 ** self.nf1 * torch.exp(-self.nf1 * (t1 - 1))
        filter2 = self.scale2 * t2 ** self.nf2 * torch.exp(-self.nf2 * (t2 - 1))
        tc = filter1 + filter2  # shape = [J, 1, K]
        if not self.individual_shapes:
            tc = torch.cat([tc[:1].repeat(14, 1, 1)])
        return torch.flip(tc, dims=(-1,))


class Encoder(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 neurons: int,
                 frames: int,
                 temporal_kernel_size: int,
                 zero_padding: Tuple[int, int],
                 nonlinearity: str,
                 input_noise: float,
                 data_has_noise: bool,
                 output_noise: float,
                 shape: Optional[str],
                 individual_shapes: bool,
                 data_covariance: torch.Tensor,
                 temporal_filter_type: Optional[str],
                 fix_first_two_centers: bool):

        super().__init__()
        self.kernel_size = kernel_size
        self.D = kernel_size * kernel_size
        self.J = neurons
        self.T = frames
        self.K = temporal_kernel_size
        self.padding_left, self.padding_right = zero_padding
        self.nonlinearity = nonlinearity
        self.data_has_noise = data_has_noise
        self.input_noise = input_noise
        self.output_noise = output_noise
        self.shape = shape
        self.temporal_filter_type = temporal_filter_type
        self.fix_first_two_centers = fix_first_two_centers


        self.register_buffer("data_covariance", data_covariance, persistent=False)

        self.diff_gaussian_weight = None

        if shape is not None:
            kernel_x = torch.rand(self.J) * (kernel_size - 1) / 2.0 + (kernel_size - 1) / 4.0
            kernel_y = torch.rand(self.J) * (kernel_size - 1) / 2.0 + (kernel_size - 1) / 4.0
            kernel_x[:2].fill_((kernel_size - 1) / 2.0)
            kernel_y[:2].fill_((kernel_size - 1) / 2.0)
            self.kernel_centers = nn.Parameter(torch.stack([kernel_x, kernel_y], dim=1))

            def zero_first_two(grad):
                grad = grad.clone()
                grad[:2, :] = 0
                return grad

            if fix_first_two_centers:
                self.kernel_centers.register_hook(zero_first_two)

            assert self.J % 2 == 0, "only even numbers are allowed for 'neurons'"
            self.register_buffer("kernel_polarities", torch.tensor([-1, 1] * (self.J // 2)))
            shape_module = get_shape_module(shape)

            self.shape_function = shape_module(kernel_size, self.J if individual_shapes else 1)

        else:
            W = 0.02 * torch.randn(self.D, self.J)
            self.W = nn.Parameter(W / W.norm(dim=0, keepdim=True))  # spatial kernel, [D, J]

        self.logA = nn.Parameter(0.02 * torch.randn(self.J))  # gain of the nonlinearity
        self.logB = nn.Parameter(0.02 * torch.randn(self.J) - 1)  # bias of the nonlinearity

        if self.K > 1:
            if self.temporal_filter_type == 'difference-of-exponentials':
                self.convolution_kernel_shape = DiffExponentialShape(self.J, self.K, individual_shapes)

            else:  # no parameterization assumption for temporal kernels
                self.convolution_kernel = nn.Parameter(0.02 * torch.randn(self.J, 1, self.K))
        else:
            self.padding_left = 0
            self.padding_right = 0
            self.convolution_kernel = None

    def kernel_variance(self):
        W = self.W / self.W.norm(dim=0, keepdim=True)
        W = W.reshape(1, self.kernel_size, self.kernel_size, self.J).mean(dim=0)
        Wx = W.pow(2).sum(dim=1)
        Wy = W.pow(2).sum(dim=0)

        coordsX = torch.arange(self.kernel_size, dtype=torch.float32, device=W.device)[:, None]
        meanWx = torch.sum(coordsX * Wx, dim=0)
        varWx = torch.sum((coordsX - meanWx).pow(2) * Wx, dim=0)
        coordsY = torch.arange(self.kernel_size, dtype=torch.float32, device=W.device)[:, None]
        meanWy = torch.sum(coordsY * Wy, dim=0)
        varWy = torch.sum((coordsY - meanWy).pow(2) * Wy, dim=0)

        return (varWx + varWy).mean()

    def kernel_similarity(self, threshold):
        if self.temporal_filter_type is None:
            spatiotemporal = (self.W.T[:, None, :] * self.convolution_kernel[:, 0, :, None]).reshape(self.J, -1)
        else:
            spatiotemporal = (self.W.T[:, None, :] * self.convolution_kernel_shape()[:, 0, :, None]).reshape(self.J, -1)
        spatiotemporal = spatiotemporal / torch.linalg.norm(spatiotemporal, dim=-1, keepdim=True)
        innerproduct = (spatiotemporal @ spatiotemporal.T).triu(1)
        return (innerproduct - threshold).relu().mean()

    def jitter_kernels(self, power=1.0):
        if hasattr(self, 'shape_function') and isinstance(self.shape_function, Shape):
            # drift the kernel centers by the power
            center = radius = (self.kernel_size - 1) / 2.0
            with torch.no_grad():
                for i in range(self.J):
                    tries = 0
                    while True:
                        drifted = self.kernel_centers[i] + power * torch.randn_like(self.kernel_centers[i])
                        if ((drifted[0] - center) ** 2 + (drifted[1] - center) ** 2).item() <= radius ** 2:
                            self.kernel_centers[i] = drifted
                            break
                        tries += 1
                        if tries >= 10:
                            r = torch.rand([]) ** 0.5
                            theta = 2 * np.pi * torch.rand([])
                            self.kernel_centers[i, 0] = center + r * theta.cos()
                            self.kernel_centers[i, 1] = center + r * theta.sin()
                            break

        else:
            with torch.no_grad():
                self.W.mul_(self.W.abs().pow(power))
                self.normalize()

    def spatiotemporal(self, input: torch.Tensor):
        # compute Z in VAE note page 8. (linear projection after the spatiotemporal kernel)
        # input.shape = [*, L, D]  (L = length of the input frames + paddings)
        y = input @ self.W # W.shape = [*, D, J], y.shape = [*, L, J]

        if self.K > 1:
            y = y.transpose(-1, -2)  # y.shape = [*, J, L]
            y = F.pad(y, [self.padding_left, self.padding_right])  # y.shape = [*, J, T + K -1]
            if self.temporal_filter_type is None:
                convolution_kernel = self.convolution_kernel
                convolution_kernel = convolution_kernel / convolution_kernel.norm(dim=-1, keepdim=True)
            else:
                self.convolution_kernel = self.convolution_kernel_shape()
                convolution_kernel = self.convolution_kernel / self.convolution_kernel.norm(dim=-1, keepdim=True)
            # convolution_kernel.shape = [J, 1, K]
            y = F.conv1d(y, convolution_kernel, groups=self.J)  # y.shape = [*, J, T]
            y = y.transpose(-1, -2)  # y.shape = [*, T, J]

        return y

    def matrix_spatiotemporal(self, input: torch.Tensor, gain: torch.Tensor):
        # compute C_rx in VAE note page 8.
        # input.shape = [LD, LD], gain.shape = [1 or B, T or 1, J]
        assert input.ndim == 2 and input.shape[0] == input.shape[1]
        L = input.shape[0] // self.D
        D = self.D

        x = input.reshape(L * D, L, D)         # shape = [LD, L, D]
        x = self.spatiotemporal(x)             # shape = [D, 1, J] or [LD, T, J]
        x = x.permute(1, 2, 0)                 # shape = [1, J, D] or [T, J, LD]
        x = x.reshape(-1, L, D)                # shape = [J, 1, D] or [TJ, L, D]
        output_dim = x.shape[0]
        x = self.spatiotemporal(x)             # shape = [J, 1, J] or [TJ, T, J]
        x = x.flatten(start_dim=1)             # shape = [J, J]    or [TJ, TJ]

        G = gain.reshape(-1, output_dim)       # shape = [1 or B, J] or [1 or B, TJ]
        x = G[:, :, None] * x * G[:, None, :]  # shape = [1 or B, J, J] or [1 or B, TJ, TJ]

        return x  # this is C_rx

    def normalize(self):
        with torch.no_grad():
            self.W /= self.W.norm(dim=0, keepdim=True)

    def forward(self, image: torch.Tensor):
        D = self.D
        L = image.shape[1]  # = T if zero-padding, T+K-1 if data-padding

        if self.shape is not None:
            self.W = self.shape_function(self.kernel_centers, self.kernel_polarities)

        gain = self.logA.exp()  # shape = [J]
        bias = self.logB.exp()
        nx = self.input_noise * torch.randn_like(image)
        if self.data_has_noise:
            y = self.spatiotemporal(image)
        else:
            y = self.spatiotemporal(image + nx)
        nr = self.output_noise * torch.randn_like(y)
        z = gain * (y - bias) + nr  # z.shape = [B, T, J]

        if self.nonlinearity == "relu":
            r = gain * (y - bias).relu()
            grad = ((y - bias) > 0).float()  # shape = [B, T, J]
        elif self.nonlinearity == "softplus":
            r = gain * F.softplus(y - bias, beta=2.5)
            grad = torch.sigmoid(2.5 * (y-bias))
        elif self.nonlinearity == "linear":
            r = gain * (y - bias)
            grad = 1.0
        elif self.nonlinearity == "absolute":
            r = (gain * (y - bias)).abs()
            grad = 1.0

        gain = gain * grad  # shape = [B, T, J]
        C_nx = self.input_noise ** 2 * torch.eye(L * D, device=image.device)
        C_zx = self.matrix_spatiotemporal(C_nx, gain)
        # shape = [1 or B, J, J] or [1 or B, TJ, TJ]

        assert C_zx.shape[1] == C_zx.shape[2]  # A or TA
        C_nr = self.output_noise ** 2 * torch.eye(C_zx.shape[-1], device=image.device)
        C_zx += C_nr

        C_zx_efficient = None

        C_z = self.matrix_spatiotemporal(self.data_covariance + C_nx, gain)
        C_z += C_nr
        return z, r, C_z, C_zx, C_zx_efficient



@dataclass
class OutputMetrics(object):
    KL: torch.Tensor = None
    loss: torch.Tensor = None
    linear_penalty: torch.Tensor = None
    quadratic_penalty: torch.Tensor = None

    def final_loss(self):
        return self.loss.mean() + self.linear_penalty + self.quadratic_penalty


class OutputTerms(object):
    logdet_numerator: torch.Tensor = None
    logdet_denominator: torch.Tensor = None
    logdet_denominator_eff: torch.Tensor = None

    r_minus_one_squared = None

    z: torch.Tensor = None
    r: torch.Tensor = None

    def __init__(self, model: "Retina"):
        self.model = model

    def calculate_metrics(self, i) -> "OutputMetrics":
        KL = self.logdet_numerator - self.logdet_denominator

        if self.logdet_denominator_eff is not None:
            KL = self.logdet_numerator - self.logdet_denominator_eff

        target = os.environ.get("FIRING_RATE_TARGET", "1")
        if 'i' in target:
            target = eval(target)
        else:
            target = float(target)

        h = self.r.sub(target).mean(dim=0)  # the equality constraint
        linear_penalty = (self.model.Lambda[:self.r.shape[-1]] * h).sum()
        quadratic_penalty = self.model.rho / 2 * (h ** 2).sum()

        return OutputMetrics(
            KL=KL,
            loss=self.model.beta * KL,
            linear_penalty=linear_penalty,
            quadratic_penalty=quadratic_penalty
        )


class Retina(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 neurons: int,
                 frames: int,
                 temporal_kernel_size: int,
                 zero_padding: Tuple[int, int],
                 input_noise: float,
                 data_has_noise: bool,
                 output_noise: float,
                 nonlinearity: str,
                 shape: Optional[str],
                 individual_shapes: bool,
                 data_covariance: torch.Tensor,
                 beta: float,
                 rho: float,
                 temporal_filter_type: Optional[str],
                 fix_first_two_centers: bool):

        super().__init__()
        self.beta = beta
        self.rho = rho
        self.D = kernel_size * kernel_size

        assert nonlinearity in {"relu", "softplus", "linear", "absolute"}

        self.encoder = Encoder(kernel_size, neurons, frames, temporal_kernel_size,
                               zero_padding, nonlinearity, input_noise, data_has_noise, output_noise, shape,
                               individual_shapes, data_covariance, temporal_filter_type, fix_first_two_centers)

        self.Lambda = nn.Parameter(torch.rand(neurons))

    def forward(self, x) -> OutputTerms:
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.D)  # x.shape = [B, L, D] (L: input time points)
        o = OutputTerms(self)
        o.z, o.r, numerator, denominator, denominator_eff = self.encoder(x)

        if numerator is not None:
            L_numerator = numerator.cholesky()
            o.logdet_numerator = 2 * L_numerator.diagonal(dim1=-1, dim2=-2).log2().sum(dim=-1)

        if denominator is not None:
            L_denominator = denominator.cholesky()
            o.logdet_denominator = 2 * L_denominator.diagonal(dim1=-1, dim2=-2).log2().sum(dim=-1)

        if denominator_eff is not None:
            L_denominator_eff = denominator_eff.cholesky()
            o.logdet_denominator_eff = 2 * L_denominator_eff.diagonal(dim1=-1, dim2=-2).log2().sum(dim=-1)

        return o