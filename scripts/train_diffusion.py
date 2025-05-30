# pylint: disable=unused-variable
import os
import copy

from sacred import Experiment

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

from chemeleon_dng.datamodule import DataModule
from chemeleon_dng.script_util import create_diffusion_module

ex = Experiment("chemeleon-rl")

ex.add_config("configs/base.yaml")
ex.add_config("configs/train_diffusion.yaml")


@ex.named_config
def csp():
    task = "csp"
    model = {
        "pred_atom_types": False,
        "cond_dim": 0,
    }


@ex.named_config
def dng():
    task = "dng"
    model = {
        "pred_atom_types": True,
        "cond_dim": 0,
    }


@ex.automain
def main(_config):
    cfg = copy.deepcopy(_config)
    print(cfg)
    pl.seed_everything(cfg["random_seed"])

    # Set up DataModule
    dm = DataModule(**cfg["datamodule"])

    # Set up DiffusionModule
    diffusion_module = create_diffusion_module(
        task=cfg["task"],
        model_configs=cfg["model"],
        optimizer_configs=cfg["optimizer"],
        **cfg["diffusion_module"],
    )

    # Set up Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/total_loss",
        save_top_k=1,
        save_last=True,
        mode="min",
        filename="best-{epoch}",
    )
    lr_callback = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="val/total_loss",
        patience=cfg["optimizer"]["early_stopping"],
        verbose=True,
        mode="min",
    )
    callbacks = [
        checkpoint_callback,
        lr_callback,
        early_stop_callback,
    ]

    # Set up Logger
    os.environ["WANDB_DIR"] = str(cfg["logger"]["save_dir"])
    if cfg["logger"]["offline"]:
        cfg["logger"]["log_model"] = False
    if cfg["logger"]["group"] is None:
        cfg["logger"][
            "group"
        ] = f"chemeleon/{cfg['task']}/{cfg['datamodule']['dataset_type']}"
    logger = WandbLogger(**cfg["logger"])

    # Set up Trainer
    trainer = pl.Trainer(
        **cfg["trainer"],
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    trainer.fit(diffusion_module, datamodule=dm, ckpt_path=cfg["resume_from"])
