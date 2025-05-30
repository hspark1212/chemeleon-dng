# https://github.com/jiaor17/DiffCSP
import enum
import math
from collections import namedtuple

import torch
import torch.nn as nn

from torch_geometric.utils import dense_to_sparse

from chemeleon_dng.scatter import scatter_mean


class CSPNetTask(enum.Enum):
    """
    Which task of the model.
    """

    CSP = enum.auto()  # Crystal Structure Prediction
    DNG = enum.auto()  # De Novo Generation
    GUIDE = enum.auto()  # Guiding Generation
    CLIP = enum.auto()  # Contrastive Learning


CSPNET_OUTPUTS = namedtuple(
    "DECODER_OUTPUTS", ["atom_types", "lattices", "frac_coords", "node_features"]
)


class SinusoidalTimeEmbeddings(nn.Module):
    """Attention is all you need."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SinusoidsEmbedding(nn.Module):
    "Embedding for periodic distance features."

    def __init__(self, n_frequencies=10, n_space=3):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.n_space = n_space
        self.frequencies = 2 * math.pi * torch.arange(self.n_frequencies)
        self.dim = self.n_frequencies * 2 * self.n_space

    def forward(self, x):
        emb = x.unsqueeze(-1) * self.frequencies[None, None, :].to(x.device)
        emb = emb.reshape(-1, self.n_frequencies * self.n_space)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()


class FilmLayer(nn.Module):
    """FiLM layer for efficient incorporation of conditional embeddings."""

    def __init__(
        self,
        hidden_dim,
        cond_dim,
        act_fn=nn.SiLU(),
    ):
        super(FilmLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.act_fn = act_fn
        self.mlp_cond = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            act_fn,
        )
        # block
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, *, x, cond_embeds):
        assert cond_embeds is not None
        cond_emb = self.mlp_cond(cond_embeds)
        scale, shift = cond_emb.chunk(2, dim=1)
        # residual block
        x_init = x.clone()
        x = self.proj(x)
        x = self.norm(x)
        x = x * scale + shift
        x = self.act_fn(x)
        x = x + x_init
        return x


class CSPLayer(nn.Module):
    """Message passing layer for cspnet."""

    def __init__(
        self, hidden_dim=128, act_fn=nn.SiLU(), dis_emb=None, ln=False, ip=True
    ):
        super(CSPLayer, self).__init__()

        self.dis_dim = 3
        self.dis_emb = dis_emb
        self.ip = ip
        if dis_emb is not None:
            self.dis_dim = dis_emb.dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 9 + self.dis_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
        self,
        node_features,
        frac_coords,
        lattices,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        hi, hj = node_features[edge_index[0]], node_features[edge_index[1]]
        if frac_diff is None:
            xi, xj = frac_coords[edge_index[0]], frac_coords[edge_index[1]]
            frac_diff = (xj - xi) % 1.0
        if self.dis_emb is not None:
            frac_diff = self.dis_emb(frac_diff)  # fourier transform
        if self.ip:
            lattice_ips = lattices @ lattices.transpose(-1, -2)
        else:
            lattice_ips = lattices
        lattice_ips_flatten = lattice_ips.view(-1, 9)
        lattice_ips_flatten_edges = lattice_ips_flatten[edge2graph]
        edges_input = torch.cat([hi, hj, lattice_ips_flatten_edges, frac_diff], dim=1)
        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(self, node_features, edge_features, edge_index):
        agg = scatter_mean(
            edge_features,
            edge_index[0],
            dim=0,
            dim_size=node_features.shape[0],
        )
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(
        self,
        *,
        node_features,
        lattices,
        frac_coords,
        edge_index,
        edge2graph,
        frac_diff=None,
    ):
        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)
        edge_features = self.edge_model(
            node_features, frac_coords, lattices, edge_index, edge2graph, frac_diff
        )
        node_output = self.node_model(node_features, edge_features, edge_index)
        return node_input + node_output


class CSPNet(nn.Module):
    """
    CSPNet model, adopted from DiffCSP

    - edge_style = fc
    - Task
    (1) CSP: pred_atom_types = False
    (2) DNG: pred_atom_types = True
    (3) Guide: cond_dim != 0
    (4) CLIP: time_dim = 0, cond_dim = 0
    """

    def __init__(
        self,
        hidden_dim=512,
        time_dim=128,
        num_layers=6,
        max_atoms=100,
        act_fn="silu",
        dis_emb="sin",
        num_freqs=128,
        ln=True,
        ip=True,
        smooth=False,
        cond_dim=0,
        pred_atom_types=True,
        task: CSPNetTask | None = None,
    ):
        super().__init__()
        self.task = task
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.pred_atom_types = pred_atom_types

        # Validate the configuration matches the task
        if task is not None:
            self._validate_task(task, time_dim, cond_dim, pred_atom_types)

        # Time embeddings
        if time_dim > 0:
            self.time_embedding = SinusoidalTimeEmbeddings(time_dim)
        else:
            self.time_embedding = None

        # Node embedding
        if smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)

        # Atom latent embedding (node + time embedding)
        if time_dim > 0:
            self.atom_latent_embedding = nn.Linear(hidden_dim + time_dim, hidden_dim)

        # Set up FiLM layer if time or condition embeddings are used
        if cond_dim > 0:
            self.film_layer = FilmLayer(hidden_dim, cond_dim)
        else:
            self.film_layer = None

        # Set up activation function
        if act_fn == "silu":
            self.act_fn = nn.SiLU()

        # Set up distance embedding
        if dis_emb == "sin":
            self.dis_emb = SinusoidsEmbedding(n_frequencies=num_freqs)
        elif dis_emb == "none":
            self.dis_emb = None

        # Set up layers
        self.num_layers = num_layers
        self.ln = ln
        self.ip = ip
        for i in range(0, num_layers):
            self.add_module(
                f"csp_layer_{i}",
                CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip),
            )
        if ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_atom_types:
            self.type_out = nn.Linear(hidden_dim, max_atoms)
        self.coord_out = nn.Linear(hidden_dim, 3, bias=False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias=False)

    def _validate_task(
        self, task: CSPNetTask, time_dim: int, cond_dim: int, pred_atom_types: bool
    ):
        if task == CSPNetTask.CSP:
            if pred_atom_types:
                raise ValueError("CSP task requires pred_atom_types = False")
        elif task == CSPNetTask.DNG:
            if not pred_atom_types:
                raise ValueError("DNG task requires pred_atom_types = True")
        elif task == CSPNetTask.GUIDE:
            if cond_dim <= 0 or not pred_atom_types:
                raise ValueError(
                    "GUIDE task requires cond_dim > 0 and pred_atom_types = True"
                )
        elif task == CSPNetTask.CLIP:
            if time_dim != 0 or cond_dim != 0 or pred_atom_types:
                raise ValueError(
                    "CLIP task requires time_dim = 0, cond_dim = 0, and pred_atom_types = False"
                )
        else:
            raise ValueError(f"Unsupported task: {task}")

    def gen_edges(self, num_atoms, frac_coords):
        # edge_style = fc
        lis = [torch.ones(n, n, device=num_atoms.device) for n in num_atoms]
        fc_graph = torch.block_diag(*lis)
        fc_edges, _ = dense_to_sparse(fc_graph)
        return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]) % 1.0

    def forward(
        self,
        *,
        atom_types,
        lattices,
        frac_coords,
        num_atoms,
        batch_idx,
        t=None,  # time embeddings can be omitted when training contrastive learning
        cond_embeds=None,
    ) -> CSPNET_OUTPUTS:
        edges, frac_diff = self.gen_edges(num_atoms, frac_coords)
        edge2graph = batch_idx[edges[0]]
        node_features = self.node_embedding(atom_types)  # [B_n, hidden_dim]

        if t is not None:
            assert self.time_embedding is not None, "Time embeddings are not defined."
            time_embeds = self.time_embedding(t)  # [B, time_dim]
            time_embeds_per_atom = time_embeds.repeat_interleave(
                num_atoms, dim=0
            )  # [B_n, time_dim]
            # Concatenate time embeddings with node features
            node_features = torch.cat(
                [node_features, time_embeds_per_atom], dim=1
            )  # [B_n, hidden_dim + time_dim]
            node_features = self.atom_latent_embedding(
                node_features
            )  # [B_n, hidden_dim]

        if cond_embeds is not None:
            assert self.film_layer is not None, "FiLM layer is not defined."
            cond_embeds_per_atom = cond_embeds.repeat_interleave(
                num_atoms, dim=0
            )  # [B_n, cond_dim]
        else:
            cond_embeds_per_atom = None

        for i in range(0, self.num_layers):
            if self.film_layer is not None:
                node_features = self.film_layer(
                    x=node_features, cond_embeds=cond_embeds_per_atom
                )
            node_features = self._modules[f"csp_layer_{i}"](
                node_features=node_features,
                lattices=lattices,
                frac_coords=frac_coords,
                edge_index=edges,
                edge2graph=edge2graph,
                frac_diff=frac_diff,
            )  # type: ignore

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_out = self.coord_out(node_features)

        graph_features = scatter_mean(node_features, batch_idx, dim=0)
        lattice_out = self.lattice_out(graph_features)
        lattice_out = lattice_out.view(-1, 3, 3)
        if self.ip:
            lattice_out = torch.einsum("bij,bjk->bik", lattice_out, lattices)
        if self.pred_atom_types:
            type_out = self.type_out(node_features)
        else:
            type_out = atom_types

        return CSPNET_OUTPUTS(
            atom_types=type_out,  # [B_n, max_atoms]
            lattices=lattice_out,  # [B, 3, 3]
            frac_coords=coord_out,  # [B_n, 3]
            node_features=node_features,  # [B_n, hidden_dim]
        )
