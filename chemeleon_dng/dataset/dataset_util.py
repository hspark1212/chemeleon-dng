from ase import Atoms
from pymatgen.core import Structure, Lattice

import torch
from torch_geometric.data import Data, Batch


def pmg_structure_to_pyg_data(pmg_structure: Structure, **kwargs) -> Data:
    return Data(
        pos=torch.tensor(pmg_structure.frac_coords, dtype=torch.float),
        atom_types=torch.tensor(pmg_structure.atomic_numbers, dtype=torch.long),
        frac_coords=torch.tensor(pmg_structure.frac_coords, dtype=torch.float),
        cart_coords=torch.tensor(pmg_structure.cart_coords, dtype=torch.float),
        lattices=torch.tensor(
            pmg_structure.lattice.matrix, dtype=torch.float
        ).unsqueeze(0),
        num_atoms=torch.tensor([len(pmg_structure)]),
        **kwargs,
    )


def batch_to_atoms_list(batch: Batch, frac_coords: bool = True) -> list[Atoms]:
    atoms_list = []
    for data in batch.to_data_list():
        atoms = Atoms(
            numbers=data.atom_types.detach().cpu().numpy(),
            cell=data.lattices.squeeze(0).detach().cpu().numpy(),
            pbc=True,
        )
        if frac_coords:
            positions = data.frac_coords.detach().cpu().numpy()
            atoms.set_scaled_positions(positions)
        else:
            positions = data.cart_coords.detach().cpu().numpy()
            atoms.set_positions(positions)
        atoms_list.append(atoms)
    return atoms_list


def batch_to_structure_list(batch: Batch, frac_coords: bool = True) -> list[Structure]:
    structure_list = []
    for data in batch.to_data_list():
        # Get atomic numbers (convert to list for pymatgen Structure)
        atomic_numbers = data.atom_types.detach().cpu().numpy().tolist()

        # Create lattice from lattice matrix
        lattice = Lattice(data.lattices.squeeze(0).detach().cpu().numpy())

        if frac_coords:
            coords = data.frac_coords.detach().cpu().numpy()
            structure = Structure(
                lattice=Lattice.from_parameters(*lattice.parameters),
                species=atomic_numbers,
                coords=coords,
                coords_are_cartesian=False,
            )
        else:
            coords = data.cart_coords.detach().cpu().numpy()
            structure = Structure(
                lattice=Lattice.from_parameters(*lattice.parameters),
                species=atomic_numbers,
                coords=coords,
                coords_are_cartesian=True,
            )
        structure_list.append(structure)
    return structure_list
