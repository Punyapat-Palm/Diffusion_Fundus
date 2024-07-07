import torch
import pytorch_lightning as pl
from data import DiffSet  # Ensure you have the necessary imports for DiffSet
from model import DiffusionModel
from config import config
from torch.utils.data import DataLoader
import imageio
import glob
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def train_model(config: dict) -> None:
    # loading model
    pass_version = None
    last_checkpoint = None
    if config['continue_train']:
        pass_version = config["version"]
        checkpoint_dir = f"./lightning_logs/{config["dataset"]}_Stage{config['diagnosis']}/version_{config['version']}/checkpoints/"
        last_checkpoint = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))[-1]
        
    # Create datasets and data loaders
    train_dataset = DiffSet(True)
    val_dataset = DiffSet(False)

    train_loader = DataLoader(
        train_dataset, batch_size=8, num_workers=4, shuffle=False, persistent_workers=True  
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, num_workers=4, shuffle=False, persistent_workers=True  
    )

    # Create model and trainer
    if config['continue_train']:
        model = DiffusionModel.load_from_checkpoint(
            last_checkpoint,
            in_size=train_dataset.size * train_dataset.size,
            t_range=config["diffusion_steps"],
            img_depth=train_dataset.depth,
        )
    else:
        model = DiffusionModel(train_dataset.size * train_dataset.size, config["diffusion_steps"], train_dataset.depth)

    # Load Trainer model
    tb_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/",
        name=f'{config["dataset"]}_Stage{config['diagnosis']}',
        version=pass_version,
    )

    trainer = pl.Trainer(max_epochs=config["max_epoch"], log_every_n_steps=7, logger=tb_logger)
    print(f'The result is saved in {trainer.logger.log_dir}')
    # Train model
    trainer.fit(model, train_loader, val_loader, ckpt_path=last_checkpoint if config['continue_train'] else None)

if __name__ == "__main__":
    train_model(config)