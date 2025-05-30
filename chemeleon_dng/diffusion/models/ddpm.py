import torch
import torch.nn.functional as F
from torch import Tensor

from chemeleon_dng.diffusion.models.base import DiffusionModelBase


class DDPM(DiffusionModelBase):
    """
    Denoising Diffusion Probabilistic Model (DDPM).
    """

    def __init__(
        self,
        *,
        betas: torch.Tensor,
    ):
        betas = torch.cat((torch.tensor([0.0]), betas))
        assert len(betas.shape) == 1, "betas must be 1-D"

        self.num_timesteps = int(betas.shape[0])

        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Coefficients for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Coefficients for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = torch.zeros_like(betas)
        self.posterior_variance[1:] = (
            betas[1:]
            * (1.0 - self.alphas_cumprod[:-1])
            / (1.0 - self.alphas_cumprod[1:])
        )
        # DDPM Coefficients
        self.ddpm_posterior_mean_coef1 = torch.ones_like(betas)
        self.ddpm_posterior_mean_coef1[1:] = 1.0 / torch.sqrt(self.alphas[1:])
        self.ddpm_posterior_mean_coef2 = torch.zeros_like(betas)
        self.ddpm_posterior_mean_coef2[1:] = (1.0 - self.alphas[1:]) / torch.sqrt(
            1 - self.alphas_cumprod[1:]
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the forward process q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps.
        :param noise: if specified, the split-out normal noise.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    @torch.no_grad()
    def p_sample(self, model_output: Tensor, x_t: Tensor, t: Tensor):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model_output: the model output, which is either the predicted epsilon.
        :param x_t: the current tensor at x_{t}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        """
        noise = torch.randn_like(x_t)
        nonzero_mask = (
            (t != 1).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )  # no noise when t == 1

        c0 = _extract_into_tensor(self.ddpm_posterior_mean_coef1, t, x_t.shape)
        c1 = _extract_into_tensor(self.ddpm_posterior_mean_coef2, t, x_t.shape)
        posterior_std = _extract_into_tensor(
            torch.sqrt(self.posterior_variance), t, x_t.shape
        )
        pred_sample = (
            c0 * (x_t - c1 * model_output) + nonzero_mask * noise * posterior_std
        )
        return pred_sample

    def training_loss(
        self,
        *,
        model_output: Tensor,
        noise: Tensor,
    ):
        loss = F.mse_loss(model_output, noise)  # Model mean type = epsilon
        return {"loss": loss}

    def p_mean_and_var(self, model_output: Tensor, x_t: Tensor, t: Tensor):
        """
        Compute the mean and variance of the posterior distribution.

        :param model_output: the model output, which is either the predicted epsilon.
        :param x_t: the current tensor at x_{t}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        """
        c0 = _extract_into_tensor(self.ddpm_posterior_mean_coef1, t, x_t.shape)
        c1 = _extract_into_tensor(self.ddpm_posterior_mean_coef2, t, x_t.shape)
        var = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        mean = c0 * (x_t - c1 * model_output)
        return mean, var


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    arr = arr.to(timesteps.device)
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
