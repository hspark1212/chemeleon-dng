from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule

from chemeleon_dng.dataset.mp_dataset import MPDataset


class DataModule(LightningDataModule):
    """DataModule for ChemeleonRL."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        dataset_type: str = "mp",
        target_condition: str | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        val_batch_size: int | None = None,
        test_batch_size: int | None = None,
    ):
        super().__init__()
        # Configs for dataset
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.target_condition = target_condition
        print(f"Data directory: {self.data_dir}")

        # Configs for dataloader
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_batch_size = batch_size if val_batch_size is None else val_batch_size
        self.test_batch_size = (
            batch_size if test_batch_size is None else test_batch_size
        )

        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    @property
    def dataset_cls(self):
        """Return the dataset class based on the dataset type."""
        if self.dataset_type == "mp":
            return MPDataset
        elif self.dataset_type == "mp_20":
            return MPDataset
        elif self.dataset_type == "mp_alex_20":
            return MPDataset
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_cls(
                data_dir=self.data_dir,
                split="train",
                target_condition=self.target_condition,
            )
            self.val_dataset = self.dataset_cls(
                data_dir=self.data_dir,
                split="val",
                target_condition=self.target_condition,
            )
        if stage == "test" or stage is None:
            self.test_dataset = self.dataset_cls(
                data_dir=self.data_dir,
                split="test",
                target_condition=self.target_condition,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Train dataset is not initialized.")
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError("Validation dataset is not initialized.")
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise ValueError("Test dataset is not initialized.")
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
