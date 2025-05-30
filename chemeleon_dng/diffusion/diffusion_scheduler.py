import math
import numpy as np
import torch


def uniform_sample_t(num_timesteps, batch_size):
    """Generate a uniform sample of timesteps.
    minimum timestep is 1, maximum timestep is num_timesteps
    """
    ts = np.random.choice(np.arange(1, num_timesteps + 1), batch_size)
    return torch.from_numpy(ts)


def get_named_beta_schedule(
    schedule_name,
    num_diffusion_timesteps,
    beta_start=0.0001,
    beta_end=0.02,
):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        return torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
    elif schedule_name == "quadratic":
        return (
            torch.linspace(beta_start**0.5, beta_end**0.5, num_diffusion_timesteps) ** 2
        )
    elif schedule_name == "cosine":
        return cosine_beta_schedule(num_diffusion_timesteps)
    elif schedule_name == "reciprocal":
        return reciprocal_beta_schedule(num_diffusion_timesteps)
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def reciprocal_beta_schedule(timesteps):
    t = torch.arange(1, timesteps + 1)
    beta_recip = 1 / (timesteps - t + 1)
    return beta_recip
