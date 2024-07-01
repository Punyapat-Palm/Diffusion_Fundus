import torch
from data import DiffSet
import pytorch_lightning as pl
from model import DiffusionModel
from torch.utils.data import DataLoader
import imageio
import glob
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def sample_images(model, train_dataset, output_dir) -> None:
    sample_batch_size = 9  # Number of images to generate in each sampling step

    # Generate random noise
    x = torch.randn(
        (sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size)
    )
    sample_steps = torch.arange(model.t_range - 1, 0, -1)

    # Denoise the initial noise for T steps
    for t in tqdm(sample_steps, desc="Sampling"):
        x = model.denoise_sample(x, t)

    # Get the final image after denoising
    final_sample = (x.clamp(-1, 1) + 1) / 2
    final_sample = (final_sample * 255).type(torch.uint8)
    final_sample = final_sample.permute(0, 2, 3, 1).cpu().numpy()  # Convert to HWC format

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save individual final images
    for i in range(final_sample.shape[0]):
        img = Image.fromarray(final_sample[i])
        img.save(os.path.join(output_dir, f"final_sample_{i:02d}.png"))

    # Create a grid of images
    grid_size = 3
    img_size = final_sample.shape[1]
    grid_img = Image.new('RGB', (img_size * grid_size, img_size * grid_size))

    for i in range(sample_batch_size):
        img = Image.fromarray(final_sample[i])
        grid_x = (i % grid_size) * img_size
        grid_y = (i // grid_size) * img_size
        grid_img.paste(img, (grid_x, grid_y))

    grid_img.save(os.path.join(output_dir, "final_grid.png"))

def train_model(config: dict) -> None:
    # Code for optionally loading model
    pass_version = None
    last_checkpoint = None

    if config['load_model']:
        pass_version = config["load_version_num"]
        last_checkpoint = glob.glob(
            f"./lightning_logs/{config['dataset']}/version_{config['load_version_num']}/checkpoints/*.ckpt"
        )[-1]

    # Create datasets and data loaders
    train_dataset = DiffSet(True, config["dataset"])
    val_dataset = DiffSet(False, config["dataset"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], num_workers=4, shuffle=False, persistent_workers=True  
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], num_workers=4, shuffle=False, persistent_workers=True  
    )

    # Create model and trainer
    if config['load_model']:
        model = DiffusionModel.load_from_checkpoint(
            last_checkpoint,
            in_size=train_dataset.size * train_dataset.size,
            t_range=config["diffusion_steps"],
            img_depth=train_dataset.depth,
        )
    else:
        model = DiffusionModel(
            train_dataset.size * train_dataset.size,
            config["diffusion_steps"],
            train_dataset.depth,
        )

    # Load Trainer model
    tb_logger = pl.loggers.TensorBoardLogger(
        "lightning_logs/",
        name=config["dataset"],
        version=pass_version,
    )

    trainer = pl.Trainer(max_epochs=config["max_epoch"], log_every_n_steps=7, logger=tb_logger)

    # Train model
    trainer.fit(model, train_loader, val_loader)

    return model, train_dataset, trainer.logger.log_dir

def get_config() -> dict:
    return {
        "diffusion_steps": 1000,
        "dataset": r"dataset",
        "max_epoch": 10,
        "batch_size": 8,
        "load_model": False,
        "load_version_num": 1,
    }

if __name__ == "__main__":   
    config = get_config()
    model, train_ds, output_dir = train_model(config)
    sample_images(model, train_ds, output_dir)