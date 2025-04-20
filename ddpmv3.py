import os
import json
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, utils  # Added utils for image grid logging

# Import custom modules
from utils import *
from modules import UNet, EMA

# Import FID computation library
from pytorch_fid import fid_score

# ------------------ Reproducibility Setup ------------------
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ------------------ Logging Setup ------------------
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")


# -------------- Diffusion Process --------------
class Diffusion:
    def __init__(self, noise_steps=1000, img_size=128, device="cuda"):
        self.noise_steps = noise_steps
        self.img_size = img_size
        self.device = device
        
        # Compute beta schedule using a cosine schedule
        self.beta = self.cosine_beta_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def cosine_beta_schedule(self, s=0.008):
        # Use float32 precision for stability
        t = torch.linspace(0, self.noise_steps, steps=self.noise_steps + 1, dtype=torch.float32)
        alpha_bar = torch.cos(((t / self.noise_steps) + s) / (1 + s) * (math.pi / 2)) ** 2
        betas = torch.clip(1 - (alpha_bar[1:] / alpha_bar[:-1]), 0.0001, 0.02)
        return betas

    def noise_images(self, x, t):
        # Ensure float32 computations and proper reshaping for broadcasting
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t].to(torch.float32))[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t].to(torch.float32))[:, None, None, None]
        # Generate noise using random normal distribution
        noise = torch.randn_like(x, dtype=torch.float32)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, batch_size):
        # Randomly sample timesteps for each image in a batch
        return torch.randint(0, self.noise_steps, (batch_size,), device=self.device, dtype=torch.int64)

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), dtype=torch.float32).to(self.device)
            # Reverse diffusion loop
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = torch.full((n,), i, device=self.device, dtype=torch.int64)
                predicted_noise = model(x, t)
                
                alpha = self.alpha[t][:, None, None, None].to(torch.float32)
                alpha_hat = self.alpha_hat[t][:, None, None, None].to(torch.float32)
                beta = self.beta[t][:, None, None, None].to(torch.float32)
                # Add noise only when not on the last step
                noise = torch.randn_like(x, dtype=torch.float32) if i > 1 else torch.zeros_like(x, dtype=torch.float32)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        # Post-process images: clamp, rescale and convert to uint8
        x = (x.clamp(-1, 1) + 1) / 2  
        x = (x * 255).type(torch.uint8)
        return x

# -------------- Dataset Setup --------------
class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, subset_size=3000):
        super(CustomDataset, self).__init__(root, transform)
        self.subset_size = subset_size
        # Limit dataset size for quicker experiments or debugging
        self.samples = self.samples[:self.subset_size]  
    
    def __len__(self):
        return self.subset_size

def get_data(args):
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = CustomDataset(root=args.dataset_path, transform=transform, subset_size=args.subset_size)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# -------------- Validation Setup --------------
def get_validation_data(args):
    """Load validation data from a separate folder."""
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = datasets.ImageFolder(root=args.val_dataset_path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

def validate(args, model, diffusion, device, val_loader):
    model.eval()
    total_loss = 0.0
    count = 0
    # Use 'mean' reduction so per-pixel average is computed, similar to training
    mse = nn.MSELoss()  
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            total_loss += loss.item()
            count += 1  # count per image
    avg_loss = total_loss / count if count > 0 else 0
    model.train()
    return avg_loss


# -------------------- FID Helper Functions --------------------
def compute_fid(real_folder, fake_folder, batch_size, device, dims=2048):
    """
    Computes the FID score using the pytorch-fid library.
    """
    fid_value = fid_score.calculate_fid_given_paths(
        [os.path.abspath(real_folder), os.path.abspath(fake_folder)],
        batch_size=batch_size,
        device=device,
        dims=dims
    )
    return fid_value

def save_images_to_folder(images, folder, prefix="img"):
    """
    Saves each image in the batch to the specified folder.
    """
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        file_path = os.path.join(folder, f"{prefix}_{i}.png")
        # Convert to float tensor in [0, 1] before saving
        utils.save_image(img.float() / 255.0, file_path)

def compute_additional_metrics(real_folder, fake_folder, args):
    """
    Computes FID (and placeholder for LPIPS) given directories with real and fake images.
    """
    fid_value = compute_fid(real_folder, fake_folder, batch_size=args.batch_size, device=args.device)
    return {"FID": fid_value, "LPIPS": 0}  

# -------------- Training Loop --------------
def train(args):
    setup_logging(args.run_name)
    os.makedirs(os.path.join("results", args.run_name), exist_ok=True)
    os.makedirs(os.path.join("models", args.run_name), exist_ok=True)
    log_file = os.path.join("results", args.run_name, "training_log.json")

    device = args.device
    dataloader = get_data(args)
    val_loader = get_validation_data(args)  # Load Validation Data

    model = UNet(c_in=3, c_out=3).to(device)
    ema_model = UNet(c_in=3, c_out=3).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()

    ema_helper = EMA(beta=0.999)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))

    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join("models", args.run_name, "ckpt.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logging.info(f"Resuming training from epoch {start_epoch}...")
        else:
            logging.warning("Checkpoint not found. Starting from scratch.")

    # Folder for fake images used in FID computation
    fake_folder = os.path.join("results", args.run_name, "fid_fake")
    os.makedirs(fake_folder, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        # Training Loop
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_helper.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("Loss/MSE", loss.item(), global_step=epoch * len(dataloader) + i)

        # Validation Loop
        val_loss = validate(args, model, diffusion, device, val_loader)
        logger.add_scalar("Loss/Val_MSE", val_loss, epoch)
        logging.info(f"Epoch {epoch}: Validation MSE = {val_loss:.6f}")

        # Generate sample images at the end of epoch
        sampled_images = diffusion.sample(ema_model, n=images.shape[0])
        # Save image grid (for visual monitoring)
        grid_path = os.path.join("results", args.run_name, f"{epoch}.jpg")
        save_images(sampled_images, grid_path)

        # Save individual generated images for FID evaluation.
        save_images_to_folder(sampled_images, fake_folder, prefix=f"epoch_{epoch}")

        # Log generated images as a grid on TensorBoard
        grid = utils.make_grid(sampled_images.float() / 255, nrow=4, normalize=True)
        logger.add_image("Generated", grid, epoch)

        # Compute additional metrics (FID and LPIPS)
        metrics = compute_additional_metrics(args.real_folder, fake_folder, args)
        logger.add_scalar("FID", metrics["FID"], epoch)
        logger.add_scalar("LPIPS", metrics["LPIPS"], epoch)
        logging.info(f"Epoch {epoch}: FID = {metrics['FID']}")

        # Append training progress in a log file (including Validation loss and FID)
        log_data = {
            "epoch": epoch,
            "mse": loss.item(),
            "val_mse": val_loss,
            "fid": metrics["FID"],
            "image": f"{epoch}.jpg"
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(log_data) + "\n")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_model_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, os.path.join("models", args.run_name, f"ckpt.pt"))

# -------------- Main and Argument Parsing --------------
def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--real_folder", type=str, default=r"fid_data/real",
                        help="Folder with real images for FID computation")
    parser.add_argument("--val_dataset_path", type=str, default=r"fid_data/val",
                        help="Folder with validation images for computing validation loss")
    # Experiment configuration arguments
    parser.add_argument("--run_name", type=str, default="DDPM_Unconditional")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--dataset_path", type=str, default=r"C:\Users\ewanp\OneDrive\Desktop\project 10\dataset\img_align_celeba")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--subset_size", type=int, default=3000)

    args = parser.parse_args()

    logging.info(f"Dataset Path: {args.dataset_path}")
    logging.info(f"Files in Dataset Path: {os.listdir(args.dataset_path)}")

    train(args)

if __name__ == '__main__':
    launch()
