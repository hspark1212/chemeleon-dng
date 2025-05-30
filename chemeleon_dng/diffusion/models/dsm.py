import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

from chemeleon_dng.diffusion.models.base import DiffusionModelBase


class DSM(DiffusionModelBase):
    """
    Denoising Score Matching (DSM).
    """

    def __init__(
        self,
        *,
        num_timesteps: int,
        sigma_begin: float = 0.005,
        sigma_end: float = 0.5,
    ):
        self.timesteps = num_timesteps
        self.sigma_begin = sigma_begin
        self.sigma_end = sigma_end
        _sigmas = torch.FloatTensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_timesteps))
        )
        _sigmas_norm = get_sigma_norm(_sigmas)

        self.sigmas = torch.cat([torch.zeros([1]), _sigmas], dim=0)
        self.sigmas_norm = torch.cat([torch.ones([1]), _sigmas_norm], dim=0)

    def q_sample(self, x_start: Tensor, t: Tensor, batch_idx: Tensor, noise=None):
        """
        Sample from the forward process q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param batch_idx: the batch index.
        :param noise: if specified, the split-out normal noise.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        sigmas = self.sigmas.to(t.device)
        sigma_t = sigmas[t]
        sigma_t_per_atom = sigma_t[batch_idx][:, None]  # [B_n, 1]
        return (x_start + sigma_t_per_atom * noise) % 1.0

    def p_sample(
        self,
        model_output: Tensor,
        x_t: Tensor,
        t: Tensor,
        batch_idx: Tensor,
        noise: Tensor | None = None,
        mode: str = "predictor",
        step_lr: float = 1e-5,
    ):
        sigmas = self.sigmas.to(t.device)
        sigmas_norm = self.sigmas_norm.to(t.device)
        sigma_t = sigmas[t]
        sigma_norm_t = sigmas_norm[t]
        if noise is None:
            noise = torch.randn_like(x_t)
        nonzero_mask = (
            (t[batch_idx] != 1).float().unsqueeze(1).expand(-1, x_t.shape[1])
        )  # no noise when t == 1
        assert nonzero_mask.shape == x_t.shape
        if mode == "predictor":
            adjacent_sigma = sigmas[t - 1]
            step_size = sigma_t**2 - adjacent_sigma**2
            std = torch.sqrt(
                (adjacent_sigma**2 * (sigma_t**2 - adjacent_sigma**2)) / (sigma_t**2)
            )
            # broadcast to [B_n, 1]
            step_size = step_size[batch_idx][:, None]  # [B_n, 1]
            std = std[batch_idx][:, None]  # [B_n, 1]
            sigma_norm_t = sigma_norm_t[batch_idx][:, None]  # [B_n, 1]

            pred_sample = (
                x_t
                - step_size * model_output * torch.sqrt(sigma_norm_t)
                + std * noise * nonzero_mask
            )
            return pred_sample % 1.0  # x_t_minus_05
        elif mode == "corrector":  # x_t = x_t_minus_05
            step_size = step_lr * (sigma_t / self.sigma_begin) ** 2
            std = torch.sqrt(2 * step_size)
            # Broadcast to [B_n, 1]
            step_size = step_size[batch_idx][:, None]  # [B_n, 1]
            std = std[batch_idx][:, None]  # [B_n, 1]
            sigma_norm_t = sigma_norm_t[batch_idx][:, None]  # [B_n, 1]

            pred_sample = (
                x_t
                - step_size * model_output * torch.sqrt(sigma_norm_t)
                + std * noise * nonzero_mask
            )
            return pred_sample % 1.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def training_loss(
        self,
        model_output: Tensor,
        noise: Tensor,
        t: Tensor,
        batch_idx: Tensor,
    ):
        sigmas = self.sigmas.to(t.device)
        sigmas_norm = self.sigmas_norm.to(t.device)
        sigma_t = sigmas[t]
        sigma_norm_t = sigmas_norm[t]
        sigma_t_per_atom = sigma_t[batch_idx][:, None]  # [B_n, 1]
        sigma_norm_t_per_atom = sigma_norm_t[batch_idx][:, None]  # [B_n, 1]
        target = d_log_p_wrapped_normal(
            sigma_t_per_atom * noise, sigma_t_per_atom
        ) / torch.sqrt(sigma_norm_t_per_atom)
        loss = F.mse_loss(model_output, target)
        return {"loss": loss}

    def p_mean_and_var(
        self, model_output: Tensor, x_t: Tensor, t: Tensor, batch_idx: Tensor
    ):
        sigmas = self.sigmas.to(t.device)
        sigmas_norm = self.sigmas_norm.to(t.device)
        sigma_t = sigmas[t]
        sigma_norm_t = sigmas_norm[t]
        adjacent_sigma = sigmas[t - 1]

        step_size = sigma_t**2 - adjacent_sigma**2

        # broadcast to [B_n, 1]
        step_size = step_size[batch_idx][:, None]  # [B_n, 1]
        sigma_norm_t = sigma_norm_t[batch_idx][:, None]  # [B_n, 1]

        var = (adjacent_sigma**2 * (sigma_t**2 - adjacent_sigma**2)) / (sigma_t**2)
        var = var[batch_idx][:, None]  # [B_n, 1]
        mean = x_t - step_size * model_output * torch.sqrt(sigma_norm_t)
        return mean, var


def p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += torch.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_


def d_log_p_wrapped_normal(x, sigma, N=10, T=1.0):
    p_ = 0
    for i in range(-N, N + 1):
        p_ += (x + T * i) / sigma**2 * torch.exp(-((x + T * i) ** 2) / 2 / sigma**2)
    return p_ / p_wrapped_normal(x, sigma, N, T)


def get_sigma_norm(sigma, T=1.0, sn=10000):
    sigmas = sigma[None, :].repeat(sn, 1)
    x_sample = sigma * torch.randn_like(sigmas)
    x_sample = x_sample % T
    normal_ = d_log_p_wrapped_normal(x_sample, sigmas, T=T)
    return (normal_**2).mean(dim=0)  # type: ignore
