from abc import abstractmethod

import torch
from torch import Tensor
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from pytorch_lightning import LightningModule

from chemeleon_dng.schema import CrystalBatch


class BaseModule(LightningModule):
    """
    Base module for configuration of optimizers and logging.
    """

    def __init__(
        self,
        optimizer: str,
        lr: float,
        weight_decay: float,
        scheduler: str,
        patience: int,
        early_stopping: int,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.patience = patience
        self.early_stopping = early_stopping
        self.warmup_steps = warmup_steps

    @abstractmethod
    def calculate_loss(self, *args, **kwargs) -> dict:
        """Calculate loss for the BaseModule.

        The output should be a dictionary, the key "total_loss" must be included.
        The value of key "total_loss" will be used for backpropagation.
        The other keys will be used for logging.
        """
        raise NotImplementedError("calculate_loss is not implemented")

    def training_step(
        self, batch: CrystalBatch, batch_idx: int  # pylint: disable=unused-argument
    ) -> Tensor:
        res = self.calculate_loss(batch)
        self._log_metrics(res=res, split="train", batch_size=batch.num_graphs)
        return res["total_loss"]

    def validation_step(
        self, batch: CrystalBatch, batch_idx: int  # pylint: disable=unused-argument
    ) -> Tensor:
        res = self.calculate_loss(batch)
        self._log_metrics(res=res, split="val", batch_size=batch.num_graphs)
        return res["total_loss"]

    def configure_optimizers(self):
        return self._set_configure_optimizers()

    @torch.no_grad()
    def _log_metrics(
        self,
        res: dict,
        split: str,
        batch_size: int | None = None,
    ):
        for k, v in res.items():
            self.log(
                f"{split}/{k}",
                v,
                batch_size=batch_size,
                on_step=True if split == "train" else False,
                sync_dist=True,
                prog_bar=True,
            )

    def _set_configure_optimizers(self):
        # Set optimizer
        if self.optimizer == "adam":
            print(f"Using Adam optimizer with lr={self.lr}")
            optimizer = Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        # Get max_steps
        if self.trainer.max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
        else:
            max_steps = self.trainer.max_steps
        # Set scheduler
        if self.scheduler == "constant":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        elif self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        elif self.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", min_lr=1e-4, factor=0.6, patience=self.patience
            )
        elif self.scheduler == "linear_decay":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, max_steps)
        elif self.scheduler == "warmup_linear":
            assert self.warmup_steps > 0, "warmup_steps must be greater than 0"
            if self.warmup_steps < 1.0:
                self.warmup_steps = int(max_steps * self.warmup_steps)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1,
                total_iters=self.warmup_steps,
            )
        else:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")
        lr_scheduler = {
            "scheduler": scheduler,
            "name": "learning rate",
            "monitor": "val/total_loss",
            "frequency": self.trainer.check_val_every_n_epoch,
        }

        return ([optimizer], [lr_scheduler])
