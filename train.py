import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from tempfile import gettempdir
from typing import Optional

import fire
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from data import get_dataset, KyotoNaturalImages
from model import Retina, OutputTerms, OutputMetrics
from util import cycle, kernel_images, plot_convolution


def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')


def train(logdir: str = datetime.now().strftime(f"{gettempdir()}/%y%m%d-%H%M%S"),
          iterations: int = 500000,
          same_video_batch: bool = False,  # sampling minibatch from a single video file
          random_flip: bool = True,  # random up-and-down, left-and-right video flip after sampling the batch
          batch_size: int = 128,
          data: str = "palmer",
          kernel_size: int = 18,  # kernel shape is kernel_size * kernel_size (square shape)
          temporal_kernel_size: int = 20,
          input_padding: Optional[str] = None,  # ["data", "zero"]
          circle_masking: bool = True,  # apply circular masking to data
          neurons: int = 144,  # number of neurons, J
          frames: int = 20,  # number of frames, T
          jittering_start: Optional[int] = 20000,  # method that jitters the kernels for faster optimization
          jittering_stop: Optional[int] = 200000,
          jittering_interval: int = 500,
          jittering_power: float = 0.5,
          centering_weight: float = 0.0,  # method that helps kernels localize (not used in the NeurIPS submission)
          centering_start: int = 0,
          centering_stop: int = -1,
          input_noise: float = 0.1,  # sigma_in (standard deviation of the Gaussian input noise)
          output_noise: float = 1.0,  # sigma_out (standard deviation of the Gaussian output noise)
          nonlinearity: str = "softplus",  # ["softplus", "relu", "linear", "absolute"]
          beta: float = -0.5,  # -0.5 by default to conform to the K&S loss
          shape: Optional[str] = "difference-of-gaussian",  # "difference-of-gaussian" for Oneshape case
          individual_shapes: bool = True,  # individual size of the RFs can be different for the Oneshape case
          optimizer: str = "adam",  # can be "adam" or "sgd"
          learning_rate: float = 0.001,
          rho: float = 1,  # coefficient to the quadratic penalty term
          temporal_filter_type: Optional[str] = "difference-of-exponentials",  # if None, it learns temporal filters without parameterization
          maxgradnorm: float = 20.0,
          load_checkpoint: str = None,  # path of the checkpoint file to resume training from
          device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
          neural_type: Optional[str] = None,  # only used for FilteredVideoDataset (NeurIPS paper Fig 5)
          fix_first_two_centers: bool = False,  # only used for FilteredVideoDataset experiments (NeurIPS paper Fig 5)
          random_seed: str = 44,
          ):
    train_args = deepcopy(locals())  # keep all arguments as a dictionary
    for arg in sys.argv:
        if arg.startswith("--") and arg[2:] not in train_args:
            raise ValueError(f"Unknown argument: {arg}")

    set_seed(seed=random_seed)

    print(f"Logging to {logdir}")
    writer = SummaryWriter(log_dir=logdir)
    writer.add_text("train_args", json.dumps(train_args))

    if input_padding == "zero":  # zero-pad during convolution
        data_frames = frames
        padding_left = round(0.8 * (temporal_kernel_size - 1))
        padding_right = temporal_kernel_size - 1 - padding_left
    elif input_padding == "data":  # read K-1 additional frames and no zero-pad during convolution
        data_frames = frames + temporal_kernel_size - 1
        padding_left, padding_right = 0, 0
    elif input_padding is None:  # no padding, the output becomes one frame if temporal_kernel_size == frames
        data_frames = frames
        padding_left, padding_right = 0, 0
    else:
        raise ValueError(f" Unsupported input_padding value: {input_padding}")

    group_size = batch_size if same_video_batch else None

    if data == "kyoto":
        assert frames == 1
        assert temporal_kernel_size == 1
        dataset = KyotoNaturalImages("kyoto", kernel_size, circle_masking, device=device)
    else:
        dataset = get_dataset(data, kernel_size, data_frames, circle_masking, group_size, random_flip, neural_type, input_noise)

    data_covariance = dataset.covariance()

    data_has_noise = False

    if data == "pink_tempfilter" or data == "real_tempfilter":
        data_has_noise = True
        temporal_kernel_size = 1

    data_loader = DataLoader(dataset, batch_size)
    data_iterator = cycle(data_loader)

    model_args = dict(
        kernel_size=kernel_size,
        neurons=neurons,
        frames=frames,
        temporal_kernel_size=temporal_kernel_size,
        zero_padding=(padding_left, padding_right),
        input_noise=input_noise,
        data_has_noise=data_has_noise,
        output_noise=output_noise,
        nonlinearity=nonlinearity,
        data_covariance=data_covariance,
        shape=shape,
        individual_shapes=individual_shapes,
        beta=beta,
        rho=rho,
        temporal_filter_type=temporal_filter_type,
        fix_first_two_centers=fix_first_two_centers,
    )

    model = Retina(**model_args).to(device)
    model_args["data_covariance"] = None

    H_X = data_covariance.cholesky().diag().log2().sum().item() + model.D / 2.0 * np.log2(2 * np.pi * np.e)
    print(f"H(X) = {H_X:.3f}")

    optimizer_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[optimizer]
    optimizer_kwargs = dict(lr=learning_rate)

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    all_params = [p for n, p in model.named_parameters() if p.requires_grad]
    last_iteration = 0
    if load_checkpoint is not None:
        checkpoint = torch.load(load_checkpoint)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Pickled model no longer supported for laoding")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        last_iteration = checkpoint["iteration"]

    if jittering_start is not None and jittering_stop is not None:
        assert jittering_interval > 0
        jittering_iterations = set(range(jittering_start, jittering_stop+1, jittering_interval))
    else:
        jittering_iterations = []


    model.train()
    with trange(last_iteration + 1, iterations + 1, ncols=99) as loop:
        for iteration in loop:
            if iteration in jittering_iterations:
                model.encoder.jitter_kernels(jittering_power)

            batch = next(data_iterator).to(device)

            torch.manual_seed(iteration)
            output: OutputTerms = model(batch)
            metrics: OutputMetrics = output.calculate_metrics(iteration)

            loss = metrics.final_loss()
            kernel_variance = model.encoder.kernel_variance()
            if centering_start <= iteration < centering_stop:
                loss = loss + centering_weight * kernel_variance.mean()

            optimizer.zero_grad()
            loss.backward()
            # print(model.encoder.kernel_centers[0:2].tolist())
            param_norm = torch.cat([param.data.flatten() for param in model.parameters()]).norm()
            grad_norm = torch.cat(
                [param.grad.data.flatten() for param in model.parameters() if param.grad is not None]).norm()
            model.Lambda.grad.neg_()

            if maxgradnorm:
                torch.nn.utils.clip_grad_norm_(all_params, maxgradnorm)

            optimizer.step()
            model.encoder.normalize()

            loop.set_postfix(dict(
                KL=metrics.KL.mean().item(),
                loss=loss.item()
            ))

            if iteration % 10 == 0:
                for key, value in output.__dict__.items():
                    if torch.is_tensor(value):
                        writer.add_scalar(f"terms/{key}", value.mean().item(), iteration)
                for key, value in metrics.__dict__.items():
                    if torch.is_tensor(value):
                        writer.add_scalar(f"terms/{key}", value.mean().item(), iteration)
                writer.add_scalar(f"terms/MI", H_X - metrics.loss.mean().item(), iteration)
                writer.add_scalar(f"train/grad_norm", grad_norm.item(), iteration)
                writer.add_scalar(f"train/param_norm", param_norm.item(), iteration)
                writer.add_scalar(f"train/kernel_variance", kernel_variance.item(), iteration)
                writer.add_scalar(f"train/final_loss", loss.item(), iteration)

            if iteration % 1000 == 0 or iteration == 1:
                W = model.encoder.W.detach().cpu().numpy()

                writer.add_image('kernels', kernel_images(W, kernel_size, 1), iteration)
                if model.encoder.convolution_kernel is not None:
                    plot = plot_convolution(model.encoder.convolution_kernel.detach().cpu().numpy())
                    writer.add_image("temporal_convolutions", plot, iteration)

                Lambda = model.Lambda.detach().cpu().numpy()
                writer.add_histogram("histograms/λ", Lambda, iteration, bins=100)

                r = output.r.detach().cpu().numpy().mean(-1)
                writer.add_histogram("histogram/r", r, iteration, bins=100)

                gain = model.encoder.logA.detach().exp().cpu().numpy()
                writer.add_histogram("histogram/gain", gain, iteration, bins=100)

                bias = model.encoder.logB.detach().exp().cpu().numpy()
                writer.add_histogram("histogram/bias", bias, iteration, bins=100)

                if hasattr(model.encoder, "shape_function"):
                    if isinstance(model.encoder.shape_function, nn.ModuleList):
                        a = torch.cat([shape.a for shape in model.encoder.shape_function])
                    else:
                        a = model.encoder.shape_function.a
                    writer.add_histogram("histogram/diffgaussian_a", a, iteration, bins=100)
                    if hasattr(model.encoder.shape_function, "b"):
                        if isinstance(model.encoder.shape_function, nn.ModuleList):
                            b = torch.cat([shape.b for shape in model.encoder.shape_function])
                        else:
                            b = model.encoder.shape_function.b
                        writer.add_histogram("histogram/diffgaussian_b", b, iteration, bins=100)

            if iteration % 1000 == 0:
                to_ignore = ["data_covariance"]
                to_restore = {}
                for key, value in model.encoder._buffers.items():
                    if key in to_ignore:
                        to_restore[key] = value
                        model.encoder._buffers[key] = None
                torch.save(model, f"{logdir}/model-{iteration}.pt")

                for key, value in to_restore.items():
                    model.encoder.register_buffer(key, value, persistent=False)

                torch.save(dict(
                    iteration=iteration,
                    args=train_args,
                    model_args=model_args,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict()
                ), f"{logdir}/checkpoint-{iteration}.pt")

            writer.flush()

    writer.close()


if __name__ == "__main__":
    fire.Fire(train)
