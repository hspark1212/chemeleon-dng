from collections import OrderedDict
from pydantic import BaseModel, Field, ConfigDict

from ase import Atoms
from ase.build.tools import sort

import torch
from torch import Tensor


class CrystalBatch(BaseModel):
    """
    A schema for a batch of crystal structures.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    atom_types: Tensor
    lattices: Tensor
    frac_coords: Tensor
    num_atoms: Tensor
    batch: Tensor  # batch_idx
    num_graphs: int | None = None
    num_nodes: int | None = None
    noise_atom_types: Tensor | None = None
    noise_lattices: Tensor | None = None
    noise_frac_coords: Tensor | None = None


class Trajectory(BaseModel):
    """
    A schema for a trajectory of crystal structures.
    Each crystal structure is represented as a CrystalBatch.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    total_steps: int
    container: OrderedDict[int, CrystalBatch] = Field(default_factory=OrderedDict)

    def __getitem__(self, t: int):
        return self.container[t]

    def __setitem__(self, t: int, value: CrystalBatch) -> None:
        self.container[t] = value

    def __len__(self) -> int:
        return len(self.container)

    def get_atoms(self, t: int = 0, idx: int | None = None) -> Atoms | list[Atoms]:
        trajectory_step = self.container[t]

        # If atom type is greater than 100 + 1, set it to 0
        trajectory_step.atom_types = torch.where(
            trajectory_step.atom_types <= 101, trajectory_step.atom_types, 0
        )
        split_atom_types = torch.split(
            trajectory_step.atom_types, trajectory_step.num_atoms.tolist()
        )
        split_frac_coords = torch.split(
            trajectory_step.frac_coords, trajectory_step.num_atoms.tolist()
        )

        atoms_list = []
        for i, (frac_coords, atom_types) in enumerate(
            zip(split_frac_coords, split_atom_types)
        ):
            atoms = Atoms(
                numbers=atom_types.detach().cpu().numpy(),
                cell=trajectory_step.lattices[i].detach().cpu().numpy(),
                pbc=True,
            )
            positions = frac_coords.detach().cpu().numpy()
            atoms.set_scaled_positions(positions)
            atoms_list.append(sort(atoms))
        if idx is None:
            return atoms_list
        else:
            return atoms_list[idx]

    def get_trajectory(self, idx: int | None = None):
        if idx is None:
            return [self.get_atoms(t, None) for t in range(self.total_steps + 1)]
        else:
            return [self.get_atoms(t, idx) for t in range(self.total_steps + 1)]
