import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import imageio
import glob
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import math
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def sample_images(model, train_dataset, output_dir) -> None:
    sample_batch_size = 1  # Number of images to generate in each sampling step

    # Generate random noise
    x = torch.randn(
        (sample_batch_size, train_dataset.depth, train_dataset.size, train_dataset.size)
    )
    sample_steps = torch.arange(model.t_range - 1, 0, -1)

    intermediate_images = []

    # Denoise the initial noise for T steps
    for t in tqdm(sample_steps, desc="Sampling"):
        x = model.denoise_sample(x, t)
        
        # Save intermediate images
        intermediate_sample = (x.clamp(-1, 1) + 1) / 2
        intermediate_sample = (intermediate_sample * 255).type(torch.uint8)
        intermediate_sample = intermediate_sample.permute(0, 2, 3, 1).cpu().numpy()
        intermediate_images.append(intermediate_sample[0])  # Assuming batch size is 1 for simplicity

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
    grid_size = math.ceil(sample_batch_size**.5)
    img_size = final_sample.shape[1]
    grid_img = Image.new('RGB', (img_size * grid_size, img_size * grid_size))

    for i in range(sample_batch_size):
        img = Image.fromarray(final_sample[i])
        grid_x = (i % grid_size) * img_size
        grid_y = (i // grid_size) * img_size
        grid_img.paste(img, (grid_x, grid_y))

    grid_img.save(os.path.join(output_dir, "final_grid.png"))
    print(f'The image is saved in {output_dir}')

    # Save GIF of intermediate images
    gif_images = [Image.fromarray(img) for img in intermediate_images]
    gif_path = os.path.join(output_dir, "sampling_process.gif")
    gif_images[0].save(
        gif_path,
        save_all=True,
        append_images=gif_images[1:],
        duration=100,
        loop=0
    )

if __name__ == "__main__":
    from data import DiffSet 
    from model import DiffusionModel
    from config import config

    checkpoint_dir = f"./lightning_logs/{config['dataset']}_Stage{config['diagnosis']}/version_{config['version']}/checkpoints/"
    last_checkpoint = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))[-1]
    output_dir = f"lightning_logs/{config['dataset']}_Stage{config['diagnosis']}/version_{config['version']}/Image"

    train_dataset = DiffSet(True)
    model = DiffusionModel.load_from_checkpoint(
        last_checkpoint,
        in_size=train_dataset.size * train_dataset.size,
        t_range=config["diffusion_steps"],
        img_depth=train_dataset.depth
    )
    sample_images(model, train_dataset, output_dir)
