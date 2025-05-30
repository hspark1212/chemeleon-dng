import warnings
from pathlib import Path

import pandas as pd

from pymatgen.core import Structure, Lattice

from torch.utils.data import Dataset
from torch_geometric.data import Data

from chemeleon_dng.dataset.dataset_util import pmg_structure_to_pyg_data

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


class MPDataset(Dataset):
    """
    Dataset for Materials Project data (e.g., MP-20)
    """

    def __init__(
        self, data_dir: str | Path, split: str, target_condition: str | None = None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_condition = target_condition
        self.data = pd.read_csv(self.data_dir / f"{split}.csv")
        if target_condition is not None:
            self.condition = self.data[target_condition]
        else:
            self.condition = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Data:
        row = self.data.iloc[idx]

        # Get the condition for the target
        if self.condition is not None:
            cond = self.condition[idx]
        else:
            cond = None

        # Read CIF file
        str_cif = row["cif"]
        st = Structure.from_str(str_cif, fmt="cif")

        # Niggli reduction
        reduced_st = st.get_reduced_structure()
        canonical_st = Structure(
            lattice=Lattice.from_parameters(*reduced_st.lattice.parameters),
            species=reduced_st.species,
            coords=reduced_st.frac_coords,
            coords_are_cartesian=False,
        )

        # Convert PMG structure to PYG Data
        data = pmg_structure_to_pyg_data(canonical_st, cond=cond)
        return data
