import torch
import torch.nn.functional as F
from torch import Tensor

from chemeleon_dng.diffusion.models.base import DiffusionModelBase


class D3PM(DiffusionModelBase):
    """
    Discrete Denoising Diffusion Probabilistic Models (D3PM).
    """

    def __init__(
        self,
        *,
        betas: torch.Tensor,
        max_atoms: int,
        d3pm_hybrid_coeff: float = 1.0,
    ):
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
        self.max_atoms = max_atoms
        self.hybrid_coeff = d3pm_hybrid_coeff
        self.eps = 1.0e-6

        # transition matrix for absorbing
        self.q_one_step_mats = torch.stack(
            [
                self.get_absorbing_transition_mat(t)
                for t in range(0, self.num_timesteps)
            ],
            dim=0,
        )

        # construct transition matrices for q(x_t | x_start)
        q_mat_t = self.q_one_step_mats[0]
        q_mats = [q_mat_t]
        for t in range(0, self.num_timesteps):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = q_mat_t @ self.q_one_step_mats[t]
            q_mats.append(q_mat_t)
        q_mats = torch.stack(q_mats, dim=0)
        self.q_mats = q_mats

        assert self.q_mats.shape == (
            self.num_timesteps + 1,
            self.max_atoms,
            self.max_atoms,
        )

        self.q_one_step_transposed = self.q_one_step_mats.transpose(1, 2)

    def get_absorbing_transition_mat(self, t: int):
        """Computes transition matrix for q(x_t|x_{t-1}).

        Args:
            t (int): timestep.
            max_atoms (int): maximum number of atoms (100 + 1 for dummy atom).
                Defaults to 101.

        Returns:
            Q_t: transition matrix. shape = (max_atoms, max_atoms)
        """
        # get beta at timestep t
        beta_t = self.betas[t].float()

        diag = torch.full((self.max_atoms,), (1 - beta_t).item())
        mat = torch.diag(diag, 0)
        # add beta_t at first row
        mat[:, 0] += beta_t
        return mat

    def at(
        self,
        a: Tensor,
        t_per_node: Tensor,
        x: Tensor,
    ):
        """Extract coefficients at specified timesteps t - 1 and conditioning data x.

        Args:
            a (Tensor): matrix of coefficients. [num_timesteps, max_atoms, max_atoms]
            t_per_node (Tensor): timesteps.[B_n]
            x (Tensor): atom_types. [B_n]

        Returns:
            a[t, x] (Tensor): coefficients at timesteps t and data x. [B_n, max_atoms]
        """
        a = a.to(x.device)
        bs = t_per_node.shape[0]
        t_per_node = t_per_node.reshape((bs, *[1] * (x.dim() - 1)))
        # when t_per_node == 0, use t_per_node == 1
        t_per_node = torch.where(t_per_node == 0, 1, t_per_node)
        return a[t_per_node - 1, x, :]

    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor,
        batch_idx: Tensor,
        noise: Tensor | None = None,
    ):
        """Sample from q(x_t | x_start) (i.e. add noise to the data).
        q(x_t | x_start) = Categorical(x_t ; p = x_start Q_{1...t})

        Args:
            x_start (Tensor): Data at t=0.
            t (Tensor): Timesteps. [B_n]
            noise (Tensor): Noise. [B_n, max_atoms]
            batch_idx (Tensor): Batch index. [B_n]

        Returns:
            Tensor: [B_n, max_atoms]
        """
        t_per_node = t[batch_idx]
        logits = torch.log(self.at(self.q_mats, t_per_node, x_start) + self.eps)
        if noise is None:
            noise = torch.rand(len(batch_idx), self.max_atoms).to(
                x_start.device
            )  # uniform noise
        noise = torch.clip(noise, self.eps, 1.0)
        gumbel_noise = -torch.log(-torch.log(noise))
        return torch.argmax(logits + gumbel_noise, dim=-1)

    def q_posterior_logits(
        self,
        x_start: Tensor,
        x_t: Tensor,
        t_per_node: Tensor,
        is_x_start_one_hot: bool = False,
    ):
        """Compute logits for q(x_{t-1} | x_t, x_start)."""
        if is_x_start_one_hot:
            x_start_logits = x_start.clone()
        else:
            x_start_logits = torch.log(
                torch.nn.functional.one_hot(x_start, self.max_atoms) + self.eps
            )

        assert x_start_logits.shape == x_t.shape + (self.max_atoms,), print(
            f"x_start_logits.shape: {x_start_logits.shape}, x_t.shape: {x_t.shape}"
        )

        fact1 = self.at(self.q_one_step_transposed, t_per_node, x_t)

        softmaxed = torch.softmax(x_start_logits, dim=-1)
        self.q_mats = self.q_mats.to(x_start.device)
        qmats2 = self.q_mats[t_per_node - 2]
        fact2 = torch.einsum("b...c, bcd -> b...d", softmaxed, qmats2)

        out = torch.log(fact1 + self.eps) + torch.log(fact2 + self.eps)

        t_broadcast = t_per_node.reshape((t_per_node.shape[0], *[1] * (x_t.dim())))
        return torch.where(t_broadcast == 1, x_start_logits, out)

    def categorical_kl_logits(self, logits1, logits2, eps=1.0e-6):
        """KL divergence between categorical distributions.

        Distributions parameterized by logits.

        Args:
            logits1: logits of the first distribution. Last dim is class dim.
            logits2: logits of the second distribution. Last dim is class dim.
            eps: float small number to avoid numerical issues.

        Returns:
            KL(C(logits1) || C(logits2)): shape: logits1.shape[:-1]
        """
        out = torch.softmax(logits1 + eps, dim=-1) * (
            torch.log_softmax(logits1 + eps, dim=-1)
            - torch.log_softmax(logits2 + eps, dim=-1)
        )
        return out.sum(dim=-1).mean()

    @torch.no_grad()
    def p_sample(
        self,
        pred_x_start_logits: Tensor,
        x_t: Tensor,
        t: Tensor,
        batch_idx: Tensor,
        noise: Tensor | None = None,
    ):
        t_per_node = t[batch_idx]
        pred_q_posterior_logits = self.q_posterior_logits(
            pred_x_start_logits, x_t, t_per_node, is_x_start_one_hot=True
        )

        if noise is None:
            noise = torch.rand(len(batch_idx), self.max_atoms).to(
                x_t.device
            )  # uniform noise
        noise = torch.clamp(noise, min=self.eps, max=1.0)
        # if t == 1, use x_start_logits
        nonzero_mask = (t_per_node != 1).to(x_t.dtype).view(-1, *([1] * (x_t.ndim)))
        gumbel_noise = -torch.log(-torch.log(noise))
        pred_sample = torch.argmax(
            pred_q_posterior_logits + gumbel_noise * nonzero_mask, dim=-1
        )
        return pred_sample

    def training_loss(
        self,
        *,
        pred_x_start_logits: Tensor,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        batch_idx: Tensor,
    ):
        t_per_node = t[batch_idx]
        true_q_posterior_logits = self.q_posterior_logits(x_start, x_t, t_per_node)
        pred_q_posterior_logits = self.q_posterior_logits(
            pred_x_start_logits, x_t, t_per_node, is_x_start_one_hot=True
        )
        vb_loss = self.categorical_kl_logits(
            true_q_posterior_logits, pred_q_posterior_logits
        )
        ce_loss = F.cross_entropy(pred_x_start_logits.flatten(0, -2), x_start.flatten())
        loss = vb_loss + ce_loss * self.hybrid_coeff
        return {
            "loss": loss,
            "vb_loss": vb_loss,
            "ce_loss": ce_loss,
        }
